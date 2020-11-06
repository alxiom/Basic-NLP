# multicampus-NLP
+ 딥러닝 자연어 처리 - 워드임베딩 부터 최신 기술까지

## requirements
+ python >= 3.6

## 준비
+ 기본 실습 자료
```bash
git clone https://github.com/hyoungseok/multicampus-NLP.git
cd multicampus-NLP
pip install -r requirements.txt
curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -x
mkdir data
wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt -O data/ratings.txt
wget https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData%20.csv -O data/chat.csv
```
+ (optional) [koWiki](https://drive.google.com/file/d/1viFZcVWba5jtVBm3PcbHlBBCKRldV6tn/view?usp=sharing) 텍스트

## 목차
1. Word Embedding
2. Machine Translation
3. Attention
4. Transformer
5. GPT
6. BERT
