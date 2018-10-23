#!/usr/bin/python3
import functools
import io
import typing
import json
import logging
import sys
import contextlib
from allennlp.common.util import JsonDict
import pybloof
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import concurrent.futures
import scipy.sparse
import itertools
from tqdm import tqdm
import multiprocessing as mp
import multiprocessing.connection
import gzip
import bz2
import dill
import mmh3
import lmdb

def parse_dict(d: JsonDict, *keys) -> typing.Tuple:
    return tuple([d[key] for key in keys])

def text_from_file(filename: str, strip_lines: bool = True) -> typing.Generator[str, None, None]:
    open_fns = [bz2.open, gzip.open, open]
    for open_fn in open_fns:
        with open_fn(filename, "rt", encoding="UTF-8", errors="replace") as p:
            lines = iter(p)
            if strip_lines:
                lines = (line.strip() for line in lines)

            # yield the first line
            try:
                line = next(lines)
            except OSError as e:
                def is_invalid_format_error(e: Exception) -> bool:
                    if len(e.args) == 1:
                        e = e.args[0]
                        if e == "Invalid data stream":
                            return True
                        if e.startswith("Not a gzipped file"):
                            return True
                    return False
                if is_invalid_format_error(e):
                    continue
                else:
                    raise
            except StopIteration:
                return
            yield line

            # yield the rest of the lines
            yield from lines
            return
    raise ValueError("No valid decompression method found")

def _make_bloom_filter(max_elements: int, p: float = 0.01) -> pybloof.StringBloomFilter:
    bloom_params = pybloof.bloom_calculator(max_elements, p)
    return pybloof.StringBloomFilter(bloom_params['size'], bloom_params['hashes'])

def unique_lines_from_files(filenames: typing.List[str]) -> typing.Generator[str, None, None]:
    seen = _make_bloom_filter(max_elements=14 * 1000000)
    for filename in filenames:
        for line in text_from_file(filename):
            line = line.strip()
            if line not in seen:
                seen.add(line)
                yield line

def json_from_file(filename: str) -> typing.Generator[JsonDict, None, None]:
    for line in text_from_file(filename):
        try:
            yield json.loads(line)
        except ValueError as e:
            logging.warning("Error while reading document (%s); skipping", e)

@contextlib.contextmanager
def file_or_stdout(filename: typing.Optional[str]):
    if filename is None:
        yield sys.stdout
    else:
        if filename.endswith(".gz"):
            open_fn = gzip.open
        elif filename.endswith(".bz2"):
            open_fn = bz2.open
        else:
            open_fn = open

        file = open_fn(filename, "wt", encoding="UTF-8")
        try:
            yield file
        finally:
            file.close()

class Corpus(object):
    def __init__(self, filenames: str):
        self.filenames = filenames.split("+")
        self.filenames.sort()

    def unique_lines(self) -> typing.Generator[str, None, None]:
        return (line for line in unique_lines_from_files(self.filenames) if len(line) > 0)

    def short_name(self) -> str:
        return "+".join([os.path.basename(f) for f in self.filenames])

def slices(n: int, i: typing.Iterable) -> typing.Iterable[typing.List]:
    i = iter(i)
    while True:
        s = list(itertools.islice(i, n))
        if len(s) > 0:
            yield s
        else:
            break

def tfidf_parallel_transform(tfidf: TfidfVectorizer, corpus: typing.Iterable[str]) -> scipy.sparse.csr_matrix:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        transformed_slices = executor.map(tfidf.transform, slices(1000, corpus))
        result = scipy.sparse.vstack(tqdm(transformed_slices))
    return result

def mp_map(fn, input_sequence: typing.Iterable) -> typing.Iterable:
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    def process_items():
        while True:
            item = input_queue.get()
            if item is None:    # sentinel value
                break
            item_index, item = item

            try:
                processed_item = fn(item)
            except Exception as e:
                output_queue.put((None, e))
            else:
                output_queue.put((processed_item, None))

        output_queue.close()
        output_queue.join_thread()

    processes = []
    try:
        for cpu_number in range(mp.cpu_count()):
            process = mp.Process(target=process_items)
            processes.append(process)
            process.start()

        item_count = 0
        for i, item in enumerate(input_sequence):
            input_queue.put((i, item))
            item_count += 1

        for _ in range(mp.cpu_count()):
            input_queue.put(None)

        while item_count > 0:
            processed_item, error = output_queue.get()
            if error is not None:
                raise error
            yield processed_item
            item_count -= 1
    finally:
        for process in processes:
            process.terminate()

def map_per_process(fn, input_sequence: typing.Iterable) -> typing.Iterable:
    pipeno_to_pipe: typing.Dict[int, multiprocessing.connection.Connection] = {}
    pipeno_to_process: typing.Dict[int, mp.Process] = {}

    def process_one_item(send_pipe: multiprocessing.connection.Connection, item):
        try:
            processed_item = fn(item)
        except Exception as e:
            send_pipe.send((None, e))
        else:
            send_pipe.send((processed_item, None))
        send_pipe.close()

    def yield_from_pipes(pipes: typing.List[multiprocessing.connection.Connection]):
        for pipe in pipes:
            result, error = pipe.recv()
            pipeno = pipe.fileno()
            del pipeno_to_pipe[pipeno]
            pipe.close()

            process = pipeno_to_process[pipeno]
            process.join()
            del pipeno_to_process[pipeno]

            if error is None:
                yield result
            else:
                raise error

    try:
        for item in input_sequence:
            receive_pipe, send_pipe = mp.Pipe(duplex=False)
            process = mp.Process(target = process_one_item, args=(send_pipe, item))
            pipeno_to_pipe[receive_pipe.fileno()] = receive_pipe
            pipeno_to_process[receive_pipe.fileno()] = process
            process.start()

            # read out the values
            timeout = 0 if len(pipeno_to_process) < mp.cpu_count() else None
            # If we have fewer processes going than we have CPUs, we just pick up the values
            # that are done. If we are at the process limit, we wait until one of them is done.
            ready_pipes = multiprocessing.connection.wait(pipeno_to_pipe.values(), timeout=timeout)
            yield from yield_from_pipes(ready_pipes)

        # yield the rest of the items
        while len(pipeno_to_process) > 0:
            ready_pipes = multiprocessing.connection.wait(pipeno_to_pipe.values(), timeout=None)
            yield from yield_from_pipes(ready_pipes)

    finally:
        for process in pipeno_to_process.values():
            if process.is_alive():
                process.terminate()

def map_in_chunks(fn, chunk_size: int, input_sequence: typing.Iterable) -> typing.Iterable:
    def process_chunk(chunk: typing.List) -> typing.List:
        return list(map(fn, chunk))

    processed_chunks = map_per_process(process_chunk, slices(chunk_size, input_sequence))
    for processed_chunk in processed_chunks:
        yield from processed_chunk


def memoize(exceptions: typing.Optional[typing.List] = None, version: int = 0):
    if exceptions is None:
        exceptions = []
    exception_ids = [id(o) for o in exceptions]
    class MemoizePickler(dill.Pickler):
        def persistent_id(self, obj):
            try:
                return exception_ids.index(id(obj))
            except ValueError:
                return None

    class MemoizeUnpickler(dill.Unpickler):
        def persistent_load(self, pid):
            return exceptions[pid]

    def memoize_decorator(fn: typing.Callable):
        with io.BytesIO() as buffer:
            pickler = MemoizePickler(buffer)
            pickler.dump(fn)
            if version > 0:
                pickler.dump(version)
            fn_hash = mmh3.hash(buffer.getvalue(), signed=False)

        lmbd_env = lmdb.open(
            "/tmp/memoize",
            map_size=1024 * 1024 * 1024 * 1024,
            metasync=False,
            meminit=False,
            max_dbs=0)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with io.BytesIO() as buffer:
                pickler = MemoizePickler(buffer)
                pickler.dump((args, kwargs))
                combined_hash = mmh3.hash_bytes(buffer.getvalue(), seed=fn_hash)

            # read from the db
            with lmbd_env.begin(buffers=True) as read_txn:
                r = read_txn.get(combined_hash, default=None)
                if r is not None:
                    unpickler = MemoizeUnpickler(io.BytesIO(r))
                    r = unpickler.load()
                    return r

            # if we didn't find anything, run the function and write to the db
            if r is None:
                r = fn(*args, **kwargs)
                with io.BytesIO() as buffer:
                    pickler = MemoizePickler(buffer)
                    pickler.dump(r)
                    with lmbd_env.begin(write=True) as write_txn:
                        write_txn.put(combined_hash, buffer.getbuffer(), overwrite=True)
            return r
        return inner
    return memoize_decorator
