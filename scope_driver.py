import pyvisa


class ScopeDriver:
    def __init__(self):
        self._rm = None
        self._inst = None

    def connect(self, ip: str, timeout_ms: int = 60000) -> None:
        self._rm = pyvisa.ResourceManager("@py")
        addr = f"TCPIP::{ip}::5025::SOCKET"
        self._inst = self._rm.open_resource(addr)
        self._inst.read_termination = "\n"
        self._inst.write_termination = "\n"
        self._inst.encoding = "latin-1"
        self._inst.timeout = timeout_ms

    def disconnect(self) -> None:
        if self._inst:
            self._inst.close()
            self._inst = None
        if self._rm:
            self._rm.close()
            self._rm = None

    def identify(self) -> str:
        return self.query("*IDN?")

    def reset(self) -> None:
        self.write("*RST")
        self.write("*CLS")

    def check_errors(self) -> list:
        resp = self.query("SYST:ERR:ALL?").strip()
        if resp.startswith("0,"):
            return []
        return [resp]

    def write(self, cmd: str) -> None:
        self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        return self._inst.query(cmd)

    def read_raw(self) -> bytes:
        return self._inst.read_raw()

    def read_ieee_block(self) -> bytes:
        header = self._inst.read_bytes(2)           # '#' + N
        n_digits = int(chr(header[1]))
        size_bytes = self._inst.read_bytes(n_digits)
        n_data = int(size_bytes)
        data = self._inst.read_bytes(n_data)
        try:
            self._inst.read_bytes(1)                # \n terminálás elnyelése
        except Exception:
            pass
        return header + size_bytes + data

    def set_timeout(self, ms: int) -> None:
        self._inst.timeout = ms

    def force_close(self) -> None:
        """Főszálból hívható: bezárja a VISA socketet, megszakítva bármilyen blokkoló query-t."""
        inst, rm = self._inst, self._rm
        self._inst = None
        self._rm = None
        try:
            if inst:
                inst.close()
        except Exception:
            pass
        try:
            if rm:
                rm.close()
        except Exception:
            pass
