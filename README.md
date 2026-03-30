# 신호 및 시스템 이론 📡

신호 처리, 시스템 분석, 그리고 컴퓨터 비전과의 연결고리를 탐구합니다.

## 📚 목차

- [개요](#개요)
- [핵심 개념](#-핵심-개념)
- [주요 주제](#-주요-주제)
- [연결된 프로젝트](#-연결된-프로젝트)
- [실습 가이드](#-실습-가이드)

## 개요

신호 및 시스템 이론은 디지털 신호 처리, 영상 처리, 그리고 현대 컴퓨터 비전의 기초가 됩니다. 푸리에 변환, 라플라스 변환, Z-변환 등 핵심 수학적 도구들을 통해 신호와 시스템을 분석하고 설계하는 방법을 다룹니다.

## 🎯 핵심 개념

### 기본 이론
- **신호의 분류**: 연속/이산 시간, 주기/비주기, 에너지/전력 신호
- **시스템 특성**: 선형성, 시불변성, 인과성, 안정성
- **응답 분석**: 임펄스 응답, 계단 응답, 주파수 응답

### 변환 도구
- **푸리에 변환**: 시간 영역 ↔ 주파수 영역
- **라플라스 변환**: 복소수 주파수 영역 분석
- **Z-변환**: 이산 시간 시스템 분석
- **프랙틱 변환**: 짧은 시간 푸리에 변환 (STFT)

## 📋 주요 주제

### 1. 푸리에 변환 기초 🔍

#### 연속 시간 푸리에 변환 (CTFT)

**정의**:
```
X(f) = ∫₋∞⁺∞ x(t) · e^(-j2πft) dt
x(t) = ∫₋∞⁺∞ X(f) · e^(j2πft) df
```

**주요 성질**:
- **선형성**: a·x₁(t) + b·x₂(t) ↔ a·X₁(f) + b·X₂(f)
- **시간 이동**: x(t-t₀) ↔ X(f)·e^(-j2πft₀)
- **주파수 이동**: x(t)·e^(j2πf₀t) ↔ X(f-f₀)
- **컨벌루션**: x₁(t)*x₂(t) ↔ X₁(f)·X₂(f)
- **파르세발 정리**: ∫|x(t)|²dt = ∫|X(f)|²df

**예시 신호 변환**:
- **rect(t)** ↔ **sinc(f)** (직사각형 함수)
- **exp(-at)·u(t)** (a>0) ↔ **1/(a+j2πf)** (지수 감쇠)
- **cos(2πf₀t)** ↔ **½[δ(f-f₀) + δ(f+f₀)]** (코사인)
- **delta(t)** ↔ **1** (임펄스)

#### 이산 시간 푸리에 변환 (DTFT)

**정의**:
```
X(e^(jω)) = Σₙ₌₋∞⁺∞ x[n] · e^(-jωn)
x[n] = 1/(2π) ∫₋π⁺π X(e^(jω)) · e^(jωn) dω
```

**연속 vs 이산 차이**:
- 이산: 주파수 영역이 주기적 (2π 주기)
- 연속: 주파수 영역이 비주기적
- DTFT: 이산 시간 ↔ 연속 주파수
- DFT: 이산 시간 ↔ 이산 주파수

#### 이산 푸리에 변환 (DFT)

**정의**:
```
X[k] = Σₙ₌₀ᴺ⁻¹ x[n] · e^(-j2πkn/N), k = 0,1,...,N-1
x[n] = 1/N Σₖ₌₀ᴺ⁻¹ X[k] · e^(j2πkn/N)
```

**고속 푸리에 변환 (FFT)**:
- **속도**: O(N²) → O(N log N)
- **알고리즘**: Cooley-Tukey (분할 정복)
- **구현**: NumPy, SciPy, FFTW

```python
import numpy as np

# FFT 예시
t = np.linspace(0, 1, 1000, endpoint=False)
f = 50  # 50Hz 신호
x = np.sin(2 * np.pi * f * t)

# FFT 계산
X = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), t[1]-t[0])

# 스펙트럼 plot
import matplotlib.pyplot as plt
plt.plot(freq[:len(freq)//2], np.abs(X[:len(X)//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
```

#### 주파수 응답 해석

**크기 응답**: |X(f)| - 각 주파수 성분의 강도
**위상 응답**: ∠X(f) - 각 주파수 성분의 위상

**실제 응용**:
- 오디오: 스펙트럼 분석, 이퀄라이저
- 통신: 변조/복조, 대역폭 분석
- 영상: 주파수 영역 필터링, 압축

### 2. 신호 샘플링 📊

#### 나이퀴스트 정리 (Nyquist-Shannon)

**핵심**: 대역제한 신호를 복원하기 위한 최소 샘플링 주파수
```
fs > 2 · fmax
```
- **fs**: 샘플링 주파수
- **fmax**: 신호의 최대 주파수
- **나이퀴스트 주파수**: fNyquist = fs/2

**예시**:
- 인간 청각: 20Hz~20kHz → CD 품질 44.1kHz
- 전화 음성: 300Hz~3.4kHz → 8kHz 샘플링
- 오디오 전문: 48kHz or 96kHz

#### 에일리어싱 현상 (Aliasing)

**원인**: 나이퀴스트 정리 위반 → 고주파가 저주파로 왜곡
```
fs < 2 · fmax  →  에일리어싱 발생
```

**시각적 예시**:
- 차륜이 뒤로 돌아가는 것처럼 보이는 wagon-wheel effect
- 디지털 오디오에서 고음이 저음으로 왜곡

**해결 방법**:
1. **안티앨리어싱 필터**: 샘플링 전 저역 통과 필터 적용
2. **과도 샘플링**: fs ≫ 2·fmax
3. **오버샘플링 ADC**: 고주파 성분을 먼저 높여 필터링

```python
import numpy as np
import matplotlib.pyplot as plt

# 에일리어싱 시뮬레이션
fs = 100  # 샘플링 주파수
t = np.arange(0, 1, 1/fs)

# 30Hz 신호 (정상)
f1 = 30
x1 = np.sin(2*np.pi*f1*t)

# 70Hz 신호 (에일리어싱)
f2 = 70
x2 = np.sin(2*np.pi*f2*t)

# 30Hz 신호와 동일하게 보임
print("70Hz 신호가 30Hz로 에일리어싱됨:", np.allclose(x1, x2))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, x1)
plt.title('30Hz 신호 (정상)')
plt.subplot(1, 2, 2)
plt.plot(t, x2)
plt.title('70Hz 신호 (에일리어싱됨)')
plt.tight_layout()
plt.show()
```

#### 재구성 필터링

**완전한 복원을 위한 조건**:
1. 샘플링 전 안티앨리어싱 필터
2. 샘플링 후 이상적인 저역 통과 필터 (reconstruction filter)
3. 홀더 (Zero-order, first-order 등)

#### 샘플링 전략

**과도 샘플링 (Oversampling)**:
- **장점**: 에일리어싱 제거 용이, 양자화 노이즈 감소
- **단점**: 데이터량 증가, 처리 속도 저하

**undersampling (Bandpass sampling)**:
- **원리**: 대역패스 신호를 나이퀴스트보다 낮게 샘플링
- **응용**: RF 신호 처리, 통신 시스템

### 3. LTI 시스템 분석 📈

#### 선형 시불변 시스템 (LTI)

**선형성 (Linearity)**:
```
x₁(t) → y₁(t)
x₂(t) → y₂(t)
a·x₁(t) + b·x₂(t) → a·y₁(t) + b·y₂(t)
```

**시불변성 (Time-Invariance)**:
```
x(t) → y(t)
x(t-t₀) → y(t-t₀)
```

#### 컨벌루션 연산

**시간 영역**:
```
y(t) = (x * h)(t) = ∫₋∞⁺∞ x(τ) · h(t-τ) dτ
```

**주파수 영역** (컨벌루션 정리):
```
Y(f) = X(f) · H(f)
```

**이산 컨벌루션**:
```
y[n] = Σₖ₌₋∞⁺∞ x[k] · h[n-k]
```

#### 전달 함수 (Transfer Function)

**연속 시간 (Laplace)**:
```
H(s) = Y(s)/X(s)
s = σ + jω
```

**이산 시간 (Z-transform)**:
```
H(z) = Y(z)/X(z)
```

**주파수 응답**:
- s = jω (연속)
- z = e^(jω) (이산)

#### 시스템 응답 특성

**임펄스 응답**: h(t) = δ(t) → h(t)
**계단 응답**: u(t) → ∫₀ᵗ h(τ)dτ
**주파수 응답**: |H(f)|, ∠H(f)

**안정성 판별**:
- **연속**: 모든 극점이 왼쪽 반평면 (Re(s) < 0)
- **이산**: 모든 극점이 단위원 내부 (|z| < 1)

**예시**: 1 차 로우패스 필터
```
H(s) = 1/(1 + s/ωc)
h(t) = ωc·e^(-ωc·t)·u(t)
```

#### 실제 LTI 시스템 분석

**시스템 식별**:
1. 입력/출력 데이터 수집
2. 전달 함수 추정 (System Identification)
3. 모델 검증

**예시**: 오디오 시스템 응답 측정
```python
import scipy.signal as signal

# 시스템 식별
t = np.linspace(0, 10, 1000)
input_signal = np.sin(2*np.pi*5*t)

# 시스템 응답 (예: RC 회로)
RC = 0.1
output = signal.lfilter([1], [RC, 1], input_signal)
```

### 4. Z-변환 개요 🧮

#### Z-변환 정의

**정의**:
```
X(z) = Σₙ₌₋∞⁺∞ x[n] · z^(-n)
x[n] = 1/(2πj) ∮ X(z) · z^(n-1) dz  (역변환)
```

**ROC (Region of Convergence)**:
- Z-변환이 수렴하는 z-평면 영역
- 시스템 안정성 판별 기준

#### Z-변환 기본 성질

1. **선형성**: a·x₁[n] + b·x₂[n] ↔ a·X₁(z) + b·X₂(z)
2. **시간 이동**: x[n-n₀] ↔ z^(-n₀) · X(z)
3. **주파수 이동**: a^n · x[n] ↔ X(z/a)
4. **컨벌루션**: x[n]*h[n] ↔ X(z) · H(z)
5. **초기값 정리**: x[0] = lim(z→∞) X(z)
6. **최후값 정리**: x[∞] = lim(z→1) (1-z⁻¹)X(z)

#### Z-평면 해석

**극점과 영점**:
- **영점 (Zero)**: X(z) = 0 이 되는 점
- **극점 (Pole)**: X(z) → ∞ 이 되는 점

**안정성 판별**:
```
시스템이 안정 ⇔ ROC 가 단위원을 포함
                 ⇔ 모든 극점이 단위원 내부 (|p_i| < 1)
```

#### Z-변환 활용

**디지털 필터 설계**:
- **FIR 필터**: 영점만 사용 (안정적)
- **IIR 필터**: 극점과 영점 모두 사용

**필터 유형**:
1. **저역통과 (Low-pass)**: 저주파 통과
2. **고역통과 (High-pass)**: 고주파 통과
3. **대역통과 (Band-pass)**: 특정 대역 통과
4. **대역저지 (Band-stop)**: 특정 대역 차단

```python
import scipy.signal as signal
import matplotlib.pyplot as plt

# 저역통과 필터 설계
b, a = signal.butter(4, 0.5, btype='low')  # 4 차 버터워스
w, h = signal.freqz(b, a)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(w/np.pi, 20 * np.log10(np.abs(h)))
plt.title('Magnitude Response')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (dB)')

plt.subplot(1, 2, 2)
# 극점 - 영점 플롯
z, p, k = signal.tf2zpk(b, a)
plt.scatter(p.real, p.imag, color='red', marker='x', s=100, label='Pole')
plt.scatter(z.real, z.imag, color='blue', marker='o', s=100, label='Zero')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect('equal')
plt.title('Pole-Zero Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

**전달 함수 예시**: 1 차 저역통과
```
H(z) = (1 - α)/(1 - αz⁻¹), α = e^(-ωcTs)
```

### 5. 공간 주파수 (Spatial Frequency) 🖼️

#### 기본 개념

**시간 vs 공간**:
- **시간 신호**: x(t) - 시간에 따른 변화
- **공간 신호**: f(x,y) - 공간상의 밝기 분포
- **공간 주파수**: 단위 거리당 진동 수 (cycles/pixel)

#### 2D 푸리에 변환

**연속 2D**:
```
F(u,v) = ∫∫ f(x,y) · e^(-j2π(ux+vy)) dx dy
f(x,y) = ∫∫ F(u,v) · e^(j2π(ux+vy)) du dv
```

**이산 2D (DFT)**:
```
F(u,v) = Σₓ=0^(M-1) Σᵧ=0^(N-1) f(x,y) · e^(-j2π(ux/M + vy/N))
```

**실제 응용**:
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 이미지 읽기
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 2D DFT 계산
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# 주파수 스펙트럼
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(f_transform), cmap='gray')
plt.title('Magnitude Spectrum (Unshifted)')

plt.subplot(1, 3, 3)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Shifted)')
plt.tight_layout()
plt.show()
```

#### 주파수 영역 필터링

**필터 유형**:

1. **저역통과 (Low-pass)**:
   - **목적**: 노이즈 제거, 블러링
   - **특징**: 고주파 성분 제거
   - **예시**: 가우시안 필터, 평균 필터

2. **고역통과 (High-pass)**:
   - **목적**: 엣지 검출, 선명화
   - **특징**: 저주파 성분 제거
   - **예시**: Laplacian, Sobel

3. **대역통과 (Band-pass)**:
   - **목적**: 특정 주파수 대역만 추출

**예시: Low-pass 필터**
```python
# Ideal Low-pass Filter
def ideal_lowpass(M, N, D0):
    u, v = np.ogrid[:M, :N]
    D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
    return (D <= D0).astype(float)

# Gaussian Low-pass Filter
def gaussian_lowpass(M, N, D0):
    u, v = np.ogrid[:M, :N]
    D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
    return np.exp(-(D**2)/(2*D0**2))

# 필터 적용
filter_mask = ideal_lowpass(M, N, 30)
filtered = f_shift * filter_mask
f_back = np.fft.ifftshift(filtered)
reconstructed = np.abs(np.fft.ifft2(f_back))
```

#### 이미지 처리 응용

**주요 응용**:
1. **압축**: JPEG, JPEG2000 (DCT, wavelet)
2. **노이즈 제거**: 주파수 영역 필터링
3. **복합**: 주파수 도메인에서의 특이점 제거
4. **Feature extraction**: Texture analysis

---

## 🔗 연결된 프로젝트

- **signals-and-cv**: 신호 처리와 컴퓨터 비전의 연결
- **CNN-From-Scratch-With-PyTorch**: CNN 의 신호 처리 관점
- **manim-theory-lab**: 시각화 실험

## 🛠 실습 가이드

### 추천 실험

#### 1. FFT 구현 및 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def manual_fft(x):
    """단순 FFT 구현 (교육용)"""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j*np.pi*k*n/N)
    return X

# 테스트
t = np.linspace(0, 1, 256, endpoint=False)
signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*25*t)

# FFT 비교
fft_manual = manual_fft(signal)
fft_numpy = np.fft.fft(signal)

print("Manual vs NumPy FFT:", np.allclose(fft_manual, fft_numpy))
```

#### 2. 에일리어싱 관찰 실험

```python
# 다양한 샘플링률에서 신호 관찰
for fs in [10, 20, 50, 100]:
    t = np.arange(0, 1, 1/fs)
    signal_30 = np.sin(2*np.pi*30*t)
    signal_70 = np.sin(2*np.pi*70*t)
    
    # 30Hz 와 70Hz 가 동일하게 보이면 에일리어싱 발생
```

#### 3. LTI 시스템 응답

```python
# 시스템 응답 시뮬레이션
from scipy import signal
import matplotlib.pyplot as plt

# 2 차 시스템
num = [1]
den = [1, 1, 10]
t, y = signal.step((num, den))

plt.plot(t, y)
plt.title('Step Response')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

#### 4. Z-평면 극점 - 영점 분석

```python
# 다양한 필터의 주파수 응답 비교
butter_low = signal.butter(4, 0.5, 'low')
butter_high = signal.butter(4, 0.5, 'high')

w, h_low = signal.freqz(butter_low[0], butter_low[1])
w, h_high = signal.freqz(butter_high[0], butter_high[1])
```

### 실습을 위한 코드 구조

```
signals-and-systems/
├── README.md
├── 01_fourier_transform/
│   ├── ctft_examples.py
│   ├── dtft_examples.py
│   └── fft_comparison.py
├── 02_sampling/
│   ├── nyquist_theorem.py
│   ├── aliasing_demo.py
│   └── reconstruction.py
├── 03_lti_systems/
│   ├── convolution_demo.py
│   ├── impulse_response.py
│   └── frequency_response.py
├── 04_z_transform/
│   ├── z_transform_basics.py
│   ├── pole_zero_analysis.py
│   └── digital_filter_design.py
└── 05_spatial_frequency/
    ├── 2d_fft.py
    ├── frequency_filtering.py
    └── image_processing.py
```

### 필수 라이브러리

```python
numpy          # 수학적 연산
scipy.signal   # 신호 처리 기능
matplotlib     # 시각화
opencv-python  # 이미지 처리
```

## 📖 참고 자료

### 기본 서적
- Oppenheim & Schafer, "Discrete-Time Signal Processing"
- Proakis & Manolakis, "Digital Signal Processing"
- Rabiner & Gold, "Theory and Application of Digital Signal Processing"
- Bracewell, "The Fourier Transform and Its Applications"

### 온라인 리소스
- MIT OpenCourseWare: Signals and Systems
- Coursera: Digital Signal Processing (Georgia Tech)
- DSP Tutorial: https://www.dspguide.com/

---
*마지막 업데이트: 2026-03-30*
*기존: 공간주파수 문서 작성 완료*
*추가: 푸리에 변환 기초, 신호 샘플링, LTI 시스템, Z-변환*