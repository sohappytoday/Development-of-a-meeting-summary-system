## Information
### v1과의 차이점
STT 모델을 tiny에서 large-v3로 업그레이드 (목적 : 오차율 감소)  

### 개발 단계
STT, Summarize, Post를 하나의 이미지로 생성 후, 컨테이너로 실행  

회의자가 많아질 때, 목소리가 섞여 구분하기가 힘듦 (모델 체인지)  
OpenAI를 OLlama Open Source로 변경할 예정 (무료 버전)  

---

## Model
### STT
- Fast-Whisper (Version : large-v3)  

### Summarize
- OpenAI (Version : gpt-4.1-mini)  

### Post
- Notion API  
