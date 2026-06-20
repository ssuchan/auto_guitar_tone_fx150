"""FX150.exe에서 zlib 압축 스트림을 스캔해 preset.xml 등 텍스트 리소스 추출 시도.

Qt rcc는 리소스를 qCompress(zlib) 형식으로 저장: [4B BE 원본길이][zlib stream].
zlib 헤더(0x78 ...)를 전수 탐색해 해제, XML/텍스트로 보이면 저장.
"""
import os
import re
import sys
import zlib

# FX150 공식 SW 기본 설치 경로. 다르면 첫 인자로 exe 경로를 넘기세요.
DEFAULT_EXE = r"C:\Program Files (x86)\FLAMMA\FX150\FX150.exe"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "extracted")
SPEC_OUT = os.path.join(ROOT, "spec", "preset.xml")  # 도구가 실제로 읽는 파라미터 스펙


def main():
    # cmd 콘솔(cp949 등)에서 인코딩 못 하는 문자가 있어도 크래시하지 않게.
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    exe = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXE
    os.makedirs(OUT_DIR, exist_ok=True)
    data = open(exe, "rb").read()
    print(f"exe {len(data)} bytes")

    hits = 0
    saved = 0
    spec_found = False
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
                    fn = os.path.join(OUT_DIR, f"stream_{start:08x}{'_xml' if is_xmlish else ''}.txt")
                    open(fn, "wb").write(out)
                    saved += 1
                    flag = " [XML?]" if is_xmlish else ""
                    print(f"@{start:#010x} -> {len(out)}B{flag}  head: {head[:80]!r}")
                    # 파라미터 스펙은 <resources>+paraPos 시그니처로 유일하게 식별 → 바로 저장
                    if not spec_found and b"<resources" in out and b"paraPos=" in out:
                        os.makedirs(os.path.dirname(SPEC_OUT), exist_ok=True)
                        open(SPEC_OUT, "wb").write(out)
                        spec_found = True
                        print(f"  -> 파라미터 스펙 발견, 저장: {SPEC_OUT}")
            except Exception:
                continue
    print(f"\nzlib 해제 성공 {hits}건, 텍스트 저장 {saved}건. -> {OUT_DIR}")
    if spec_found:
        print("[완료] spec/preset.xml 생성 완료 — 이제 학습을 실행할 수 있습니다.")
    else:
        print("[실패] 파라미터 스펙을 못 찾음. exe 경로가 맞는지(FX150.exe) 확인하세요.")


if __name__ == "__main__":
    main()
