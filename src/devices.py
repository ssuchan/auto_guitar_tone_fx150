"""FX150 오디오 장치 검출.

이름에 'FX150'이 들어간 캡처/재생 장치를 찾아 hostapi별로 인덱스를 반환한다.
WASAPI > WDM-KS > DirectSound > MME 순으로 우선한다 (지연시간 기준).
"""
import sounddevice as sd

HOSTAPI_PREF = ["Windows WASAPI", "Windows WDM-KS", "Windows DirectSound", "MME"]


def _hostapi_rank(name):
    for i, pref in enumerate(HOSTAPI_PREF):
        if pref.lower() in name.lower():
            return i
    return len(HOSTAPI_PREF)


def find_fx150():
    """반환: dict {'capture': (idx, info), 'playback': (idx, info)} 또는 None."""
    hostapis = sd.query_hostapis()
    capture, playback = [], []
    for idx, d in enumerate(sd.query_devices()):
        if "FX150" not in d["name"]:
            continue
        ha_name = hostapis[d["hostapi"]]["name"]
        rank = _hostapi_rank(ha_name)
        if d["max_input_channels"] > 0:
            capture.append((rank, idx, d, ha_name))
        if d["max_output_channels"] > 0:
            playback.append((rank, idx, d, ha_name))
    if not capture or not playback:
        return None
    capture.sort()
    playback.sort()
    return {
        "capture": (capture[0][1], capture[0][2], capture[0][3]),
        "playback": (playback[0][1], playback[0][2], playback[0][3]),
    }


if __name__ == "__main__":
    r = find_fx150()
    if r is None:
        print("FX150 오디오 장치를 찾지 못함. USB 연결/전원 확인.")
    else:
        ci, cd, cha = r["capture"]
        pi, pd, pha = r["playback"]
        print(f"캡처   : idx={ci}  ch={cd['max_input_channels']}  sr={int(cd['default_samplerate'])}  [{cha}]")
        print(f"재생   : idx={pi}  ch={pd['max_output_channels']}  sr={int(pd['default_samplerate'])}  [{pha}]")
