from dataclasses import dataclass
import numpy as np
from chalk import *
from colour import Color
import chalk
from dataclasses import dataclass
from typing import List, Any
from collections import Counter
from numba import cuda
import numba
import random

@dataclass
class ScalarHistory:
    last_fn: str
    inputs: list

    def __radd__(self, b):
        return self + b

    def __add__(self, b):
        if isinstance(b, (float, int)):
            return self
        if isinstance(b, Scalar):
            return ScalarHistory(self.last_fn, self.inputs + [b])
        if isinstance(b, ScalarHistory):
            return ScalarHistory(self.last_fn, self.inputs + b.inputs)
        return NotImplemented
        
class Scalar:
    def __init__(self, location):
        self.location = location

    def __mul__(self, b):
        if isinstance(b, (float, int)):
            return ScalarHistory("id", [self])
        if isinstance(b, Scalar):
            return ScalarHistory("*", [self, b])
        return NotImplemented

    def __radd__(self, b):
        return self + b
        
    def __add__(self, b):
        if isinstance(b, (float, int)):
            return ScalarHistory("id", [self])
        if isinstance(b, Scalar):
            return ScalarHistory("+", [self, b])
        if isinstance(b, ScalarHistory):
            return ScalarHistory("+", [self] + b.inputs)
        return NotImplemented
    
class Table:
    def __init__(self, name, array):
        self.name = name
        self.incoming = []
        self.array = array

        self.size = array.shape
    
    def __getitem__(self, index):
        self.array[index]
        if isinstance(index, int):
            index = (index,)
        assert len(index) == len(self.size), "Wrong number of indices"
        if index[0] >= self.size[0]:
            assert False, "bad size"

        return Scalar((self.name,) + index)

    def __setitem__(self, index, val):
        self.array[index]
        if isinstance(index, int):
            index = (index,)
        assert len(index) == len(self.size), "Wrong number of indices"
        if index[0] >= self.size[0]:
            assert False, "bad size"
        if isinstance(val, Scalar):
            val = ScalarHistory("id", [val])
        if isinstance(val, (float, int)):
            return
        assert isinstance(val, ScalarHistory), "Assigning an unrecognized value"
        self.incoming.append((index, val))

@dataclass(frozen=True, eq=True)
class Coord:
    x: int
    y: int

    def enumerate(self):
        k = 0
        for i in range(self.y):
            for j in range(self.x):
                yield k, Coord(j, i)
                k += 1

    def tuple(self):
        return (self.x, self.y)


class RefList:
    def __init__(self):
        self.refs = []
        
    def __getitem__(self, index):
        return self.refs[-1][index]

    def __setitem__(self, index, val):
        self.refs[-1][index] = val


class Shared:
    def __init__(self, cuda):
        self.cuda = cuda

    def array(self, size, ig):
        if isinstance(size, int):
            size = (size,)
        s = np.zeros(size)
        cache = Table("S" + str(len(self.cuda.caches)), s)
        # self.caches.append(cache)
        self.cuda.caches.append(RefList())
        self.cuda.caches[-1].refs = [cache]
        self.cuda.saved.append([])
        return self.cuda.caches[-1]


class Cuda:
    blockIdx: Coord
    blockDim: Coord
    threadIdx: Coord
    caches: list
    shared: Shared

    def __init__(self, blockIdx, blockDim, threadIdx):
        self.blockIdx = blockIdx
        self.blockDim = blockDim
        self.threadIdx = threadIdx
        self.caches = []
        self.shared = Shared(self)
        self.saved = []

    def syncthreads(self):
        for i, c in enumerate(self.caches):
            old_cache = c.refs[-1]
            # self_links = cache.self_links()
            # cache.clean()
            temp = old_cache.incoming
            old_cache.incoming = self.saved[i]
            self.saved[i] = temp
            cache = Table(old_cache.name + "'", old_cache.array)

            c.refs.append(cache)

    def finish(self):
        for i, c in enumerate(self.caches):
            old_cache = c.refs[-1]
            old_cache.incoming = self.saved[i]

    def rounds(self):
        if len(self.caches) > 0:
            return len(self.caches[0].refs)
        else:
            return 0


#li Some drawing constants.

black = Color("black")
white = Color("white")
im = image(
    "robot.png", "https://raw.githubusercontent.com/minitorch/diagrams/main/robot.png"
).scale_uniform_to_x(1)
colors = list(Color("red").range_to(Color("blue"), 10))

def table(name, r, c):
    if r == 0:
        return concat(
            [rectangle(1, 1).translate(0, j).named((name, j)) for j in range(c)]
        ).center_xy()
    return concat(
        [
            rectangle(1, 1).translate(i, j).named((name, i, j))
            for i in range(r)
            for j in range(c)
        ]
    ).center_xy()


def myconnect(diagram, loc, color, con, name1, name2):
    bb1 = diagram.get_subdiagram_envelope(name1)
    bb2 = diagram.get_subdiagram_envelope(name2)
    assert bb1 is not None, f"{name1}: You may be reading/writing from an un'synced array"
    assert bb2 is not None, f"{name2}: You may be reading/writing from an un'synced array"
    off = P2(loc[0] - 0.5, loc[1] - 0.5) * 0.85
    dia = empty()
    if con:
        dia += (
            arc_between(bb1.center - V2(0.5, 0), bb2.center + off, 0)
            .line_width(0.04)
            .line_color(color)
        )
    dia += place_at(
        [rectangle(0.95, 0.95).fill_opacity(0).line_color(color).line_width(0.15)],
        [bb1.center],
    )
    dia += place_at(
        [circle(0.1).line_width(0.04).fill_color(color)], [bb2.center + off]
    )
    return dia

def draw_table(tab):
    t = text(tab.name, 0.5).fill_color(black).line_width(0.0)
    if len(tab.size) == 1:
        tab = table(tab.name, 0, *tab.size)
    else:
        tab = table(tab.name, *tab.size)
    tab = tab.line_width(0.05)
    return tab.beside((t + vstrut(0.5)), -unit_y)


def draw_connect(tab, dia, loc2, color, con):
    return concat(
        [
            myconnect(dia, loc2, color, con, (tab.name,) + loc, inp.location)
            for (loc, val) in tab.incoming
            for inp in val.inputs
        ]
    )

def grid(mat, sep):
    return vcat([ hcat([y for y in x] , sep) for x in mat], sep )

def draw_base(_, a, c, out):
    inputs = vcat([draw_table(d) for d in a], 2.0).center_xy()
    shared_tables = [[draw_table(c2.refs[i]) for i in range(1, c.rounds())] for c2 in c.caches]
    shareds = grid(shared_tables, 1.0).center_xy()
    outputs = draw_table(out).center_xy()
    return hcat([inputs, shareds, outputs], 2.0)


def draw_coins(tpbx, tpby):
    return concat(
        [
            (circle(0.5).fill_color(colors[tt]).fill_opacity(0.7) + im).translate(
                pos.x * 1.1, pos.y * 1.1
            )
            for tt, pos in Coord(tpbx, tpby).enumerate()
        ]
    )
    

def label(dia, content):
    t = vstrut(0.5) / text(content, 0.5).fill_color(black).line_width(0) / vstrut(0.5)
    dia = dia.center_xy()
    return (dia + dia.juxtapose(t, -unit_y)).center_xy()


    
def draw_results(results, name, tpbx, tpby, sparse=False):
    full = empty()
    blocks = []
    locations = []
    base = draw_base(*results[Coord(0, 0)][Coord(0, 0)])
    for block, inner in results.items():
        dia = base
        for pos, (tt, a, c, out) in inner.items():
            loc = (
                pos.x / tpbx + (1 / (2 * tpbx)),
                (pos.y / tpby)
                + (1 / (2 * tpby)),
            )
            color = colors[tt]
            
            lines = True
            if sparse:
                lines = (pos.x == 0 and pos.y == 0) or (
                    pos.x == (tpbx - 1)
                    and pos.y == (tpby - 1)
                )
            all_tabs = (
                a + [c2.refs[i] for i in range(1, c.rounds()) for c2 in c.caches] + [out]
            )
            dia = dia + concat(
                draw_connect(t, dia, loc, color, lines) for t in all_tabs
            )
        height = dia.get_envelope().height

        # Label block and surround
        dia = hstrut(1) | (label(dia, f"Block {block.x} {block.y}")) | hstrut(1)
        dia = dia.center_xy().pad(1.2)
        env = dia.get_envelope()
        dia = dia + rectangle(env.width, env.height, 0.5).line_color(
            Color("grey")
        ).fill_opacity(0.0)

        
        blocks.append(dia.pad(1.1))
        locations.append(P2(block.x, block.y))

    # Grid blocks
    env = blocks[0].get_envelope()
    offset = V2(env.width, env.height)
    full = place_at(blocks, [offset * l for l in locations])

    coins = draw_coins(tpbx, tpby)

    full = (
        vstrut(1.5)
        / text(name, 1)
        / vstrut(1)
        / coins.center_xy()
        / vstrut(1)
        / full.center_xy()
    )
    full = full.pad(1.1).center_xy()
    env = full.get_envelope()
    set_svg_height(50 * env.height)


    chalk.core.set_svg_output_height(500)
    return rectangle(env.width, env.height).fill_color(white) + full


#

@dataclass
class CudaProblem:
    name: str
    fn: Any
    inputs: List[np.ndarray]
    out: np.ndarray
    args: Tuple[int] = ()
    blockspergrid: Coord = Coord(1, 1)
    threadsperblock: Coord = Coord(1, 1)
    spec: Any = None
        
    def run_cuda(self):
        fn = self.fn
        fn = fn(numba.cuda)
        jitfn = numba.cuda.jit(fn)
        jitfn[self.blockspergrid.tuple(), self.threadsperblock.tuple()](
            self.out, *self.inputs, *self.args
        )
        return self.out

    def run_python(self):
        results = {}
        fn = self.fn
        for _, block in self.blockspergrid.enumerate():
            results[block] = {}
            for tt, pos in self.threadsperblock.enumerate():
                a = []
                args = ["a", "b", "c", "d"]
                for i, inp in enumerate(self.inputs):
                    a.append(Table(args[i], inp))
                out = Table("out", self.out)

                c = Cuda(block, self.threadsperblock, pos)
                fn(c)(out, *a, *self.args)
                c.finish()
                results[block][pos] =  (tt, a, c, out)
        return results

    def score(self, results):

        total = 0
        full = Counter()
        for pos, (tt, a, c, out) in results[Coord(0, 0)].items():
            total += 1
            count = Counter()
            for out, tab in [(False, c2.refs[i]) for i in range(1, c.rounds()) for c2 in c.caches] + [(True, out)]:
                for inc in tab.incoming:
                    if out:
                        count["out_writes"] += 1
                    else:
                        count["shared_writes"] += 1
                    for ins in inc[1].inputs:
                        if ins.location[0].startswith("S"):
                            count["shared_reads"] += 1
                        else:
                            count["in_reads"] += 1
            for k in count:
                if count[k] > full[k]:
                    full[k] = count[k]
        print(f"""# {self.name}
 
   Score (Max Per Thread):
   | {'Global Reads':>13} | {'Global Writes':>13} | {'Shared Reads' :>13} | {'Shared Writes' :>13} |
   | {full['in_reads']:>13} | {full['out_writes']:>13} | {full['shared_reads']:>13} | {full['shared_writes']:>13} | 
""") 
    
    def show(self, sparse=False):
        results = self.run_python()
        self.score(results)
        return draw_results(results, self.name,
                            self.threadsperblock.x, self.threadsperblock.y, sparse)
    
    def check(self):
        x = self.run_cuda()
        y = self.spec(*self.inputs)
        try:
            np.testing.assert_allclose(x, y)
            print("Passed Tests!")
            from IPython.display import HTML
            pups = [
            "2m78jPG",
            "pn1e9TO",
            "MQCIwzT",
            "udLK6FS",
            "ZNem5o3",
            "DS2IZ6K",
            "aydRUz8",
            "MVUdQYK",
            "kLvno0p",
            "wScLiVz",
            "Z0TII8i",
            "F1SChho",
            "9hRi2jN",
            "lvzRF3W",
            "fqHxOGI",
            "1xeUYme",
            "6tVqKyM",
            "CCxZ6Wr",
            "lMW0OPQ",
            "wHVpHVG",
            "Wj2PGRl",
            "HlaTE8H",
            "k5jALH0",
            "3V37Hqr",
            "Eq2uMTA",
            "Vy9JShx",
            "g9I2ZmK",
            "Nu4RH7f",
            "sWp0Dqd",
            "bRKfspn",
            "qawCMl5",
            "2F6j2B4",
            "fiJxCVA",
            "pCAIlxD",
            "zJx2skh",
            "2Gdl1u7",
            "aJJAY4c",
            "ros6RLC",
            "DKLBJh7",
            "eyxH0Wc",
            "rJEkEw4"]
            return HTML("""
            <video alt="test" controls autoplay=1>
                <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
            </video>
            """%(random.sample(pups, 1)[0]))
            
        except AssertionError:
            print("Failed Tests.")
            print("Yours:", x)
            print("Spec :", y)
