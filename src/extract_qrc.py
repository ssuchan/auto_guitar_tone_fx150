"""FX150.exe에서 zlib 압축 스트림을 스캔해 preset.xml 등 텍스트 리소스 추출 시도.

Qt rcc는 리소스를 qCompress(zlib) 형식으로 저장: [4B BE 원본길이][zlib stream].
zlib 헤더(0x78 ...)를 전수 탐색해 해제, XML/텍스트로 보이면 저장.
"""
import zlib
import re

EXE = r"C:\Program Files (x86)\FLAMMA\FX150\FX150.exe"
OUT_DIR = r"C:\Users\USER\Desktop\auto_guitar_tone\extracted"


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    data = open(EXE, "rb").read()
    print(f"exe {len(data)} bytes")

    hits = 0
    saved = 0
    i = 0
    # zlib 헤더 후보: 0x78 0x01 / 0x78 0x9c / 0x78 0xda
    for m in re.finditer(rb"\x78[\x01\x9c\xda]", data):
        pos = m.start()
        for start in (pos, pos):  # 스트림 시작 = 헤더 위치
            try:
                d = zlib.decompressobj()
                out = d.decompress(data[start:start + 2_000_000])
                if len(out) < 40:
                    continue
                hits += 1
                # 텍스트성(출력의 대부분이 인쇄가능 ASCII) 판정
                printable = sum(1 for b in out[:4000] if 9 <= b <= 126)
                if printable / min(len(out), 4000) > 0.85:
                    head = out[:200].decode("latin-1", "replace")
                    is_xmlish = any(t in out[:2000] for t in
                                    (b"<?xml", b"<preset", b"<param", b"<module",
                                     b"<effect", b"<list", b"name=", b"<root", b"<FX"))
                    fn = f"{OUT_DIR}\\stream_{start:08x}{'_xml' if is_xmlish else ''}.txt"
                    open(fn, "wb").write(out)
                    saved += 1
                    flag = " [XML?]" if is_xmlish else ""
                    print(f"@{start:#010x} -> {len(out)}B{flag}  head: {head[:80]!r}")
            except Exception:
                continue
    print(f"\nzlib 해제 성공 {hits}건, 텍스트 저장 {saved}건. -> {OUT_DIR}")


if __name__ == "__main__":
    main()
