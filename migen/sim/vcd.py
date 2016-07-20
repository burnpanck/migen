from itertools import count
import tempfile
import os
from collections import OrderedDict
import shutil

from migen.fhdl.namer import build_namespace


def vcd_codes():
    codechars = [chr(i) for i in range(33, 127)]
    for n in count():
        q, r = divmod(n, len(codechars))
        code = codechars[r]
        while q > 0:
            q, r = divmod(q, len(codechars))
            code = codechars[r] + code
        yield code


class VCDWriter:
    def __init__(self, filename):
        self.filename = filename
        self.buffer_file = tempfile.TemporaryFile(
            dir=os.path.dirname(filename), mode="w+")
        self.codegen = vcd_codes()
        self.codes = OrderedDict()
        self.signal_values = dict()
        self.t = 0

    def _write_value(self, f, signal, value):
        l = len(signal)
        if value < 0:
            value += 2**l
        if l > 1:
            fmtstr = "b{:0" + str(l) + "b} {}\n"
        else:
            fmtstr = "{}{}\n"
        try:
            code = self.codes[signal]
        except KeyError:
            code = next(self.codegen)
            self.codes[signal] = code
        f.write(fmtstr.format(value, code))

    def set(self, signal, value):
        if (signal not in self.signal_values
                or self.signal_values[signal] != value):
            self._write_value(self.buffer_file, signal, value)
            self.signal_values[signal] = value

    def delay(self, delay):
        self.t += delay
        self.buffer_file.write("#{}\n".format(self.t))

    def close(self):
        with open(self.filename, "w") as out:
            ns = build_namespace(self.codes.keys())
            for signal, code in self.codes.items():
                name = ns.get_name(signal)
                out.write("$var wire {len} {code} {name} $end\n"
                          .format(name=name, code=code, len=len(signal)))
            out.write("$dumpvars\n")
            for signal in self.codes.keys():
                self._write_value(out, signal, signal.reset.value)
            out.write("$end\n")
            out.write("#0\n")

            self.buffer_file.seek(0)
            shutil.copyfileobj(self.buffer_file, out)
            self.buffer_file.close()


class DummyVCDWriter:
    def set(self, signal, value):
        pass

    def delay(self, delay):
        pass

    def close(self):
        pass

def VCD_events(fh):
    """ iterates change events in a VCD file
    """
    import re
    signals = {}
    mult = 0
    num_sigs = None
    hier = []
    time = 0

    re_time = re.compile(r"^#(\d+)")
    re_1b_val = re.compile(r"^([01zxZX])(.+)")
    re_Nb_val = re.compile(r"^[br](\S+)\s+(.+)")

    while True:
        line = fh.readline()
        if line == '':  # EOF
            break

        line = line.strip()

        if "$enddefinitions" in line:
            num_sigs = len(signals)
            yield signals

        elif "$timescale" in line:
            statement = line
            if not "$end" in line:
                while fh:
                    line = fh.readline()
                    statement += line
                    if "$end" in line:
                        break

            timescale = ''.join(statement[1:-1])
            mult = 1


        elif "$scope" in line:
            # assumes all on one line
            #   $scope module dff end
            hier.append(line.split()[2])  # just keep scope name

        elif "$upscope" in line:
            hier.pop()

        elif "$var" in line:
            # assumes all on one line:
            #   $var reg 1 *@ data $end
            #   $var wire 4 ) addr [3:0] $end
            ls = line.split()
            type = ls[1]
            size = ls[2]
            code = ls[3]
            name = "".join(ls[4:-1])
            path = '.'.join(hier)
            full_name = path + name
            signals.setdefault(code,[]).append({
                'type': type,
                'name': name,
                'size': size,
                'hier': path,
            })


        elif line.startswith('#'):
            re_time_match = re_time.match(line)
            time = mult * int(re_time_match.group(1))
            yield time


        elif line.startswith(('0', '1', 'x', 'z', 'b', 'r', 'Z', 'X')):
            re_1b_val_match = re_1b_val.match(line)
            re_Nb_val_match = re_Nb_val.match(line)
            if re_Nb_val_match:
                value = re_Nb_val_match.group(1)
                code = re_Nb_val_match.group(2)
            elif re_1b_val_match:
                value = re_1b_val_match.group(1)
                code = re_1b_val_match.group(2)
            else:
                raise ValueError('Unable to parse line "%s"'%line)
            yield (code, value)
