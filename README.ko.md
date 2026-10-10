# Audio-Describe-MCP

**LLM 에이전트가 소리를 "볼" 수 있게 해 주는 MCP 서버.** 오디오 파일을 모델이 읽을 수 있는 것들로 바꿉니다: 고정된 **ID**, **수치 서명**, **스펙트로그램 이미지**. 색인을 만들 필요도, 모델을 내려받을 필요도 없고, 아무것도 밖으로 나가지 않습니다.

[English](README.md)

LLM은 소리를 듣지 못합니다. 에이전트가 효과음을 구분해야 할 때("어제 만든 천둥과 새로 만든 게 같은가?", "이 휙 소리에 8kHz 위 고주파가 있나?") 볼 수 있는 게 아무것도 없습니다. 이 서버는 가리킨 파일마다 세 가지를 줍니다.

| | 에이전트가 받는 것 | 방법 |
|---|---|---|
| **ID** | `a-b6dd10e860`: 내용이 같은 복사본은 같은 ID | 파일 바이트의 SHA-1 |
| **서명** | 음량, 지배·중심 주파수, 대역별 에너지(8kHz 위 포함), attack, decay, 잡음/음정 | FFT와 포락선 (numpy / scipy) |
| **그림** | PNG: 위에 파형, 아래에 스펙트로그램, kHz·초 눈금, 8kHz 점선 | Pillow. 이미지를 읽는 모델이 이미지로 엽니다 |

선택 기능으로 `semantic=true`를 주면 Google의 EmbeddingGemma 2가 두 소리가 **같은 종류**인지도 봅니다([선택: 의미 비교](#선택-의미-비교)). 핵심 기능에는 필요 없습니다.

## 예시

합성한 천둥 같은 울림(`rumble`, [`examples/make_examples.py`](examples/make_examples.py)로 만듦)에 `audio_describe`:

```
a-b6dd10e860  rumble.wav  4.00 s, 48000 Hz, 1 ch
level: rms -25.6 dBFS, peak -7.8 dBFS | attack 160 ms, decay 2425 ms | flatness 0.558 (noise-like)
spectrum: dominant 35 Hz, centroid 189 Hz, 95% of the energy below 398 Hz
bands: <500 Hz 99% | 500-2k 1% | 2k-8k 0% | >8k 0%   (the optional embedding model only hears up to 8 kHz)
spectrogram: <cache>/a-b6dd10e860_0_4000.png   (open it as an image: waveform on top, 0-24 kHz below, dotted line = 8 kHz)
```

그 울림과 종소리 같은 소리(`ping`)에 `audio_compare`: 표는 무엇이 다른지 말해 주고, 쌓은 그림은 A를 위에 B를 아래에 같은 주파수 눈금으로 보여 줍니다.

```
notable differences: duration 4.00 s -> 2.00 s; dominant 35 Hz -> 1195 Hz; centroid 189 Hz -> 1543 Hz; flatness 0.56 -> 0.00;
<500 Hz 99% -> 0%; 500-2k 1% -> 76%; 2k-8k 0% -> 24%; attack 160 ms -> 0 ms; decay 2425 ms -> 1385 ms
```

![rumble vs ping](docs/img/compare-rumble-vs-ping.png)

8kHz 위의 소리(선택 기능인 임베딩 모델은 듣지 못하는 대역)는 서명과 그림이 맡습니다. 6kHz와 12kHz 사인파를 비교하면 `2k-8k 100% -> 0%; >8k 0% -> 100%`이고, 그림에서는 두 번째 선이 8kHz 점선 위에 나타납니다.

![6 kHz vs 12 kHz](docs/img/compare-6k-vs-12k.png)

## 도구

| 도구 | 하는 일 |
|---|---|
| `audio_describe(path, start_s=0, end_s=None, spectrogram=True)` | **아무** 오디오 파일(wav mp3 flac ogg m4a opus aac wma)의 ID, 서명, 스펙트로그램 PNG. 한 번에 최대 300초이고, 다른 구간은 `start_s` / `end_s`로 지정합니다. |
| `audio_compare(a, b, semantic=False, spectrogram=True)` | `b`는 파일 하나 또는 최대 20개의 목록. 같은 ID(내용이 같은 파일) 여부, 무엇이 다른지의 표, `b`가 하나면 두 스펙트로그램을 쌓은 PNG 한 장. `semantic=true`면 후보를 `a`와의 임베딩 유사도 순으로 정렬합니다. |

결과는 일부러 짧게 해서 토큰을 적게 씁니다. 도구는 읽기 전용이고 캐시 폴더에 PNG만 씁니다.

### 서명 읽는 법

| 항목 | 뜻 |
|---|---|
| `rms`, `peak` | dBFS 단위 음량 |
| `attack` | 포락선이 가장 큰 5ms 프레임의 3dB 안에 처음 들어올 때까지의 시간. 일정한 순음은 attack 0 |
| `decay` | 그 시점부터 포락선이 30dB 내려갈 때까지. `ms+ (still audible at the end)`면 끝까지 내려가지 않은 것 |
| `dominant`, `centroid`, `95% rolloff` | 가장 센 주파수, 스펙트럼 무게중심, 에너지의 95%가 그 아래에 있는 주파수 |
| `flatness` | 0 = 순음, 1 = 백색잡음. 소리가 실제로 차지하는 대역에서 계산. 0.2 이상 noise-like, 0.05 미만 tonal |
| `bands` | 500Hz 아래, 500~2k, 2k~8k, 8kHz 위의 에너지 비율 |

## 설치

Python 3.10 이상과 [ffmpeg](https://ffmpeg.org/)가 필요합니다(`PATH`에 두거나 `AUDIO_DESCRIBE_FFMPEG`로 지정).

```bash
pip install "git+https://github.com/ssain3d-lgtm/Audio-Describe-MCP.git"
```

먼저 명령줄로 써 보세요.

```bash
audio-describe describe 경로/sound.wav
audio-describe compare 경로/a.wav 경로/b.wav
```

### MCP 클라이언트에서 쓰기

서버는 stdio로 MCP를 씁니다. Claude Code:

```bash
claude mcp add audio-describe -- audio-describe-mcp
```

JSON 설정을 받는 클라이언트(Qwen Code `settings.json`, Claude Desktop 등)는 가상환경에 설치했다면 실행 파일의 전체 경로를 쓰세요.

```json
{
  "mcpServers": {
    "audio-describe": {
      "command": "audio-describe-mcp",
      "env": { "AUDIO_DESCRIBE_FFMPEG": "/path/to/ffmpeg" }
    }
  }
}
```

그다음 이렇게 시키면 됩니다. *"`C:\sounds\thunder.wav`를 분석하고 스펙트로그램을 열어서 음높이, 길이, 8kHz 위에 뭔가 있는지 말해줘."* 모델이 `audio_describe`를 부르고 자기 이미지 읽기 도구로 PNG를 엽니다.

설정은 전부 선택 사항인 환경변수입니다.

| 변수 | 기본값 | |
|---|---|---|
| `AUDIO_DESCRIBE_FFMPEG` | `PATH`의 `ffmpeg` | ffmpeg 경로 |
| `AUDIO_DESCRIBE_CACHE` | Windows `%LOCALAPPDATA%\audio-describe-mcp\spectro`, 그 외 `~/.cache/audio-describe-mcp/spectro` | PNG 저장 위치(300MB를 넘으면 오래된 것부터 삭제) |
| `AUDIO_DESCRIBE_MODEL` | `google/embeddinggemma-2` | 선택 비교용 모델 |
| `AUDIO_DESCRIBE_IDLE_SECONDS` | `120` | 임베딩 모델을 안 쓰면 이 시간 뒤에 해제 |
| `AUDIO_DESCRIBE_STDERR` | 로그 파일 | `inherit`면 서버의 stderr를 그대로 둡니다. 기본은 캐시 옆 `server.log`에 씁니다. 일부 클라이언트(Qwen Code)는 서버 stderr를 읽지 않아서, 로그가 쌓이면 서버가 멈추기 때문입니다. |

## 선택: 의미 비교

```bash
pip install "audio-describe-mcp[semantic] @ git+https://github.com/ssain3d-lgtm/Audio-Describe-MCP.git"
```

`audio_compare(..., semantic=true)`는 각 파일의 앞 30초를 [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)(텍스트+오디오 타워, CPU, FP32. 처음 쓸 때 약 1.5GB를 내려받음)로 임베딩해 코사인 유사도를 알려 줍니다. 처음 쓸 때 올리고, 계속 쓰는 동안 유지하고, 120초 놀면 해제합니다. 올라와 있는 동안 프로세스가 약 3GB를 쓰고(PyTorch와 모델), 여유 RAM이 3.5GB보다 적으면 올리지 않고 서명 표만 돌려줍니다.

효과음 라이브러리로 이 PC에서 측정한 결과입니다.

- **소리끼리 비교는 됩니다.** 6종류(폭발, 클릭, 천둥, 레이저, 발소리, 동전) 30개에서 가장 가까운 소리가 같은 종류인 비율이 87%(우연이면 17%), 평균 코사인은 같은 종류끼리 0.94, 다른 종류끼리 0.88. 표본이 작고 정답은 파일 이름으로 정했습니다.
- **텍스트 라벨로 소리를 맞히는 건 안 됩니다.** "thunder", "explosion" 같은 라벨과 소리의 유사도로는 6개 중 0~1개만 맞혔기 때문에 라벨 분류기는 제공하지 않습니다.
- **8kHz 위는 듣지 못합니다**(모델이 16kHz 오디오를 읽음). 그 대역은 서명과 그림이 맡습니다.

## 한계

- 에이전트는 여전히 소리를 *듣지* 못합니다. 수치와 그림으로 추론하므로 판정("건조한 버즈", "천둥 같다")은 근거 있는 추측이고, 중요하면 직접 들어 보세요.
- `audio_describe`는 한 번에 최대 300초를 분석합니다(잘렸으면 응답에 적습니다). 샘플레이트가 32kHz 미만(전화 음질)이거나 48kHz 초과인 파일에는 무엇을 담을 수 없는지 한 줄 안내가 붙습니다.
- 디코딩에 ffmpeg가 필요합니다. 선택 모델 내려받기 말고는 네트워크로 아무것도 보내지 않습니다.
- 도구는 클라이언트가 넘기는 모든 로컬 경로를 읽습니다. 파일을 맡길 수 있는 클라이언트에만 서버를 쓰세요.
- Windows 11, Python 3.12, ffmpeg 8.0.1에서 시험했습니다. 임베딩 모델을 내린 뒤 메모리를 돌려주는 선택 동작 말고는 Windows 전용 코드가 없지만 Linux와 macOS는 시험하지 못했습니다.

## 측정한 성능

Ryzen 9 9950X3D에서 측정했습니다(CPU만 사용: 모델을 `device="cpu"`로 만듭니다).

| | |
|---|---|
| 17.8초 스테레오 mp3에 `audio_describe`(PNG 포함) | 약 0.15초, 프로세스 전체가 약 100MiB(기본 설치, PyTorch 없음) |
| `audio_compare` 서명 표만 | 약 0.5초 |
| 첫 `semantic=true`(PyTorch 임포트, 모델 적재, 3개 파일) | 약 12초, 프로세스가 약 2.9GB까지 늘어남 |
| 같은 호출을 다시(모델이 이미 적재됨) | 약 4초 |
| 유휴 해제 뒤(Windows) | 작업 집합이 시작 수준으로 돌아옴(트림 직후 약 1MiB) |

## 실제 에이전트로 확인

Qwen Code가 쓰는 로컬 비전 모델(Qwen3.8 기반, [Strata](https://github.com/Niko1221/Strata)로 서빙)에게 파일 이름 대신 `tone_a` / `tone_b`로 가린 스펙트로그램 두 장을 주고, 8kHz 점선 위에 에너지가 있는 쪽을 물었습니다. 두 PNG를 열어서 정확히 답했습니다(약 12kHz 대 약 6kHz). 이미지를 읽는 다른 클라이언트도 같은 방식으로 될 것으로 보지만 시도해 보지는 않았습니다.

## 개발

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # Windows: .venv\Scripts\pip
pytest                                                         # torch가 필요한 테스트는 extra 없이는 건너뜀
pip install -e ".[semantic]" && pytest                         # 의미 비교 테스트도 실행(대역 모델을 써서 내려받기 없음)
python examples/make_examples.py                               # docs/img를 다시 만들고 예시 출력을 보여 줌
```

테스트는 합성한 소리를 직접 만들어 쓰고(오디오 파일은 포함하지 않음), 실제 서버를 띄워 MCP 클라이언트처럼 두 도구를 부르는 stdio 테스트도 있습니다.

```
src/audio_describe_mcp/
  signature.py     ffmpeg 디코드, ID, 구간, 서명, 비교 표                    (numpy / scipy)
  spectrogram.py   그림, 쌓은 그림, 캐시 정리                                (Pillow)
  core.py          Describer.describe / .compare 와 모델이 읽는 텍스트
  semantic.py      선택 기능: EmbeddingGemma 2 인코더(유휴 해제 포함)
  server.py        MCP stdio 서버(도구 2개)       cli.py   명령줄
```

## 라이선스

MIT, [LICENSE](LICENSE) 참고. 선택 모델은 Google의 EmbeddingGemma 2입니다(Hugging Face에 Apache-2.0 태그가 있고, 모델 카드는 Gemma 금지 사용 정책도 가리킵니다). 직접 내려받는 것이며 그 약관이 모델에 적용됩니다. ffmpeg는 별도 프로그램이고 별도 라이선스입니다.
