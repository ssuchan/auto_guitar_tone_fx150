# auto_guitar_tone

유튜브 일렉기타 커버 영상의 톤을 분석해서, FLAMMA FX150 멀티이펙터의 설정값을
자동으로 탐색하는 프로그램.

## 목표

`유튜브 URL` + `내 기타 DI 녹음` → `내 DI를 유튜브 톤에 가깝게 만드는 FX150 설정값`

## 동작 구조 (목표)

```
[1] 타겟 확보            [2] 리앰프 루프              [3] 비교         [4] 최적화
유튜브 오디오            후보 설정 → FX150 적용        목표 vs 결과     Bayesian opt
yt-dlp 다운로드          DI 재생 → FX150 → 녹음        perceptual      (Optuna)가
Demucs 기타 분리   ──→   = 처리된 오디오         ──→   loss 계산   ──→ 다음 후보 제안
                         ▲──────────── 루프 반복 ──────────────────────────┘
                                                              → 최종 FX150 설정값
```

## 하드웨어 조사 결과 (2026-06-16)

- 장비: FLAMMA FX150, SW: `C:\Program Files (x86)\FLAMMA\FX150`
- USB: `VID_34DB & PID_8004`, USB Composite Device (MI_00 = MEDIA 제어 인터페이스)
- 통신: SW가 **libusb 독자 프로토콜**로 장비 제어 (`libusb_bulk_transfer` 등)
- 프리셋: 고정 바이너리 구조체 `PresetPara_TypeDef`, import/export 지원, `:/preset/preset.xml`
- **USB MIDI 포트 없음** → MIDI 제어 경로 불가
- USB 오디오 양방향:
  - 캡처: `마이크 배열(FX150)` (이펙트 처리된 기타음 녹음)
  - 재생: `스피커(FX150)`

## 제어 레이어 (최대 난관) — HID로 판명

장치는 USB 컴포지트:
- MI_00 = MEDIA (USB 오디오 in/out)
- MI_03 = **HIDClass** → 제어 채널. usage_page=0x0001, product 'FX150'

MIDI(USB) 불가지만 **HID 경로 확보**. libusb 벤더 프로토콜보다 쉬움 (드라이버 교체 불필요).
`hidapi`로 read/write 가능. 남은 일 = 에디터가 노브 변경 시 보내는 HID 리포트 디코딩:
1. Wireshark + USBPcap로 에디터의 HID 출력 리포트 캡처 (PC→장치 write는 수동 hidapi로 관찰 불가)
2. 리포트 구조 디코딩 (`PresetPara_TypeDef` 매핑)
3. hidapi로 재현 → 파라미터 프로그램 제어

대안: 물리 노브를 돌려 device→host 입력 리포트를 hidapi로 관찰 (포맷이 대칭이면 단서).

## 리앰프 경로 (Phase 1 결과)

USB 재생(스피커 FX150) → 캡처(마이크 FX150)로 신호 안 돌아옴 → **USB 재생은 이펙트 우회.**
=> 리앰프는 아날로그: PC 아날로그 출력 → FX150 입력잭 → 이펙트 → USB 캡처.
녹음 절반(USB 캡처)은 확보, 입력 절반(아날로그 출력)만 케이블링 필요.

정밀 재검증(reamp_probe.py, 2026-06-16): 초기 routing_test는 재생/캡처를 별도 스트림으로
열어 캡처가 0으로 죽고 SR(48k/44.1k)도 불일치한 결함이 있었음. 단일 풀듀플렉스(sd.playrec,
SR일치) + 1kHz 톤 FFT 검출 + clean 앰프 프리셋 활성 상태로 재측정 → 톤 비율 0.0029(무음)
vs 0.0035(재생), 사실상 무변화. USB 재생이 DSP 우회함을 깨끗이 확정. 케이블 불가피.

## 단계별 계획 (검증 기준 포함)

- [x] Phase 0: 환경/장비 탐지 → 검증: FX150 오디오 in/out 인덱스 자동 검출 (devices.py)
- [x] Phase 1: 리앰프 경로 테스트 → USB 재생 이펙트 우회 확인, 아날로그 리앰프 필요 (routing_test.py)
- [x] Phase 1b: 제어 채널 식별 → HID(MI_03) 확인 (usb_probe.py, hid_probe.py)
- [x] Phase 2: HID 프로토콜 역분석 — 완료
  - [x] 프레임 구조 해독: `aa55 <len16 LE> <cmd16 LE> <payload> <crc16 BE>`
  - [x] CRC16 역추정: poly=0x1021 init=0 refin=F refout=F xorout=0xFFFF, 대상=aa55이후, BE 저장 (crc_crack.py)
  - [x] 인코더/디코더 + 캡처 3종 자기검증 통과 (fx150_protocol.py)
  - [x] 실제 전송 검증: cycle82 전송 시 FX150 화면이 모듈 슬롯 이동 → write 반영 확인
  - [x] 파라미터 맵 확보: exe의 zlib 리소스에서 preset.xml 추출 (extract_qrc.py → spec/preset.xml)
  - [x] 스펙 파서 + cmd↔체인 매핑 (fx150_spec.py). 0x93=AMP 6파라미터 캡처 일치로 검증
- [x] Phase 3: 유튜브 다운로드 + Demucs 분리 (fetch_separate.py) — 실URL 검증 완료 (35s Short → other stem 20s 추출). Demucs 파이썬 API 사용(CLI 저장은 torchcodec 의존으로 깨짐), KMP_DUPLICATE_LIB_OK로 Anaconda OpenMP 충돌 회피
- [x] Phase 4: perceptual loss (tone_loss.py) — 합성신호 검증: 동일0/음색차<밝기차<왜곡차 순서 정확
- [x] Phase 6: Optuna TPE 최적화 루프 (optimizer.py) — mock 평가자로 수렴 검증
- [x] 후보→장비 인코더 (apply_preset.py) — 캡처 AMP/FX 프레임 정확 재구성, 사람이 읽는 출력
- [x] 엔드투엔드 글루 (main.py)
- [~] Phase 5: 아날로그 리앰프 평가자 (reamp.py) — 코드 완성, **케이블 연결 후 실측 필요**

## 사용법 (실행 전 준비)
1. 케이블: PC 라인아웃 → FX150 기타 입력잭 (DI 리앰프용)
   - **FX150 USB OUTPUT 설정 = "이펙팅(effected)" 시그널** (매뉴얼 43p: 드라이/이펙팅 선택 가능).
     드라이로 두면 생기타만 캡처돼 비교 무의미. PC 출력 볼륨은 낮게(클리핑/임피던스 방지).
   - 참고: USB 플레이백(PC→FX150)은 MIX 모니터링 전용으로 이펙트 우회(매뉴얼 44p, reamp_probe.py 확정) → 디지털 리앰프 불가.
2. DI 녹음: 내 기타 클린 연주 wav
3. `python fetch_separate.py URL [start] [dur]` → work/target_guitar.wav (첫 실행 Demucs 모델 다운로드)
4. `python devices.py`로 라인아웃 장치 인덱스 확인
5. FLAMMA 에디터 닫기 (HID 점유 충돌 방지)
6. `python main.py --di my_di.wav --target work/target_guitar.wav --play-device N --trials 150`
7. 출력된 설정이 장비에 적용됨 → 마음에 들면 FX150에서 수동 저장

## 남은 실측/튜닝 (하드웨어 연결 후)
- reamp.py 경로 실측: DI 재생→FX150 처리→USB 캡처 동작 확인
- loss 가중치 튜닝 (실제 기타 톤쌍으로)
- ~~단계형 최적화(거친 모델탐색→미세 노브) 도입~~ 완료(optimizer.staged_optimize). mock 벤치 flat 74→staged 59. 실톤쌍으로 n_coarse/n_fine 비율 튜닝은 잔여
- payload 파라미터 스케일링(역방향 범위 HI CUT 등) 실측 보정

### 체인 ↔ HID cmd 맵 (Phase 2 산출물)
0x82 = 시그널 체인 슬롯 선택(화면 이동). 0x91~0x9a = 각 체인 모듈 파라미터:
| cmd | 체인 | 모델 | 기본 파라미터 |
|---|---|---|---|
| 0x91 | FX | 8 | THRESHOLD, RATIO, ATTACK, LEVEL |
| 0x92 | OD | 22 | GAIN, TONE, VOLUME |
| 0x93 | AMP | 58 | GAIN, BASS, MIDDLE, TREBLE, PRESENCE, MASTER |
| 0x94 | CAB | 80 | TUBE, LOW CUT, HI CUT, SPACE, LEVEL |
| 0x95 | FXLOOP | 2 | SEND/RETURN LEVEL |
| 0x96 | NS | 3 | THRESHOLD |
| 0x97 | EQ | 4 | 6밴드 + LEVEL |
| 0x98 | MOD | 20 | RATE, DEPTH, TONE, LEVEL |
| 0x99 | DELAY | 9 | TIME, F.BACK, SUB-D, LEVEL |
| 0x9a | REVERB | 7 | PRE DELAY, DECAY, TONE, LEVEL |

payload 구조(추정, 검증중): `[enable LE16][model LE16][param1 LE16][param2 LE16]...`
- type="0" 연속값 `min_max_step_decimals_unit`, type="1" 열거형 `OPT_OPT_...`(값=인덱스)

### HID 프로토콜 (역분석 결과)
- 프레임: `AA 55 | len16(LE)=len(cmd+payload) | cmd16(LE) | payload | crc16(BE)`
- HID 리포트(64B): `byte0=프레임길이 | 프레임 | 0패딩`, write 시 report_id 0 prefix
- CRC: `crc=0; for b: crc^=b<<8; 8x{ crc = (crc&0x8000)?((crc<<1)^0x1021):(crc<<1) }; crc^=0xFFFF`
- 관측된 cmd: 0x91~0x9a, 0x82(블록/씬?), 0xa3/0xa4/0xa6/0xbb 등. 노브 sweep은 0x92/0x93에서 확인.
- 장치는 노브 조작 시 host로 입력 리포트를 보냄(상태 동기화) → 디코드로 현재값 읽기 가능.
- [ ] Phase 3: 유튜브 다운로드 + Demucs 분리 → 검증: 기타 stem wav 생성
- [ ] Phase 4: perceptual loss → 검증: 알려진 쌍에서 거리값 합리적
- [ ] Phase 5: 리앰프 캡처 자동화 → 검증: DI 재생→처리음 녹음 반복 동작
- [ ] Phase 6: Bayesian 최적화 루프 → 검증: 반복 시 loss 감소

## 솔직한 한계

- Demucs는 일렉기타 전용 stem 없음 → "other"에 섞임, 타겟 오염
- 유튜브 기타의 기타/캐비넷/마이크/믹스가 내 것과 달라 완벽 일치 불가 → 톤 색깔 근사
- 하드웨어 실시간 렌더 → 평가 횟수 제한 (수십~수백)
- 제어 레이어(USB RE)가 막히면 자동 루프 불가, 수동 보조로 격하
