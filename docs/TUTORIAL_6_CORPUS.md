# 튜토리얼 6 : 훈련 데이터 불러오기
이번 튜토리얼은 모델을 훈련하기 위해 말뭉치(corpus)를 로드하는 내용을 다룹니다. 
이번 튜토리얼은 여러분이 라이브러리의 [기본 유형](/resources/docs/TUTORIAL_1_BASICS.md)에 익숙하다 가정하고 진행됩니다.

## 말뭉치 오브젝트
`corpus`는 모델을 훈련하는데 사용되는 데이터 세트입니다. 이는 모델 훈련 중 훈련, 검증 및 테스트 분할에 사용되는 문장들, 개발을 위한 문장 목록 및 테스트 문장 목록으로 구성됩니다.

다음 예제는 the Universal Dependency Treebank for English를 말뭉치 오브젝트로 초기화하는 코드입니다.
```python
import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
```
위 코드를 처음 실행한다면 the Universal Dependency Treebank for English를 하드디스크에 다운로드합니다.
그 다음 훈련, 테스트, 개발을 위한 `corpus`로 분할합니다. 아래 코드를 통해 각각의 `corpus`에 몇개의 문장이 들어있는지 확인할 수 있습니다.
```python
# 몇개의 문장이 train split에 있는지 출력합니다.
print(len(corpus.train))

# 몇개의 문장이 test split에 있는지 출력합니다.
print(len(corpus.test))

# 몇개의 문장이 dev split에 있는지 출력합니다.
print(len(corpus.dev))
```

각 split의 객체에 직접 접근할 수 있습니다. 아래의 코드는 test split의 처음 문장을 출력합니다 :
```python
# training split의 처음 문장을 출력합니다.
print(corpus.test[0])
```
결과입니다 : 
```console
Sentence: "What if Google Morphed Into GoogleOS ?" - 7 Tokens
```

이 문장은 통사적, 형태학적 정보가 tag되어 있습니다. POS 태그를 사용해 문장을 인쇄해보겠습니다 :
```python
# print the first Sentence in the training split
print(corpus.test[0].to_tagged_string('pos'))
```
결과입니다 : 
```console
What <WP> if <IN> Google <NNP> Morphed <VBD> Into <IN> GoogleOS <NNP> ? <.>
```
이 말뭉치는 tag되어 있고 훈련에 사용할 수 있습니다.

### 도움을 주는 함수들
`corpus`는 유용한 도움 함수들이 많이 포함되어 있습니다. `downsample()`을 호출하고 비율을 정해 데이터를 다운샘플링 할 수 있습니다. 
우선 말뭉치를 얻습니다.
```python
import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
```
그리고 말뭉치를 다운샘플링합니다.
```python
import flair.datasets
downsampled_corpus = flair.datasets.UD_ENGLISH().downsample(0.1)
```
두 말뭉치를 출력하는 것을 통해 10%를 다운 샘플링 한 것을 확인할 수 있습니다.
```python
print("--- 1 Original ---")
print(corpus)

print("--- 2 Downsampled ---")
print(downsampled_corpus)
```
결과입니다 :
```console
--- 1 Original ---
Corpus: 12543 train + 2002 dev + 2077 test sentences

--- 2 Downsampled ---
Corpus: 1255 train + 201 dev + 208 test sentences
```

### 레이블 사전 만들기
다수의 경우 예측할 레이블이 포함되어 있는 "사전"이 필요합니다. `make_label_dictionary` 메소드를 호출하고 `label_type`을 전달해 `corpus`에서 바로 사전을 만들 수 있습니다.

예를 들어, 위에서 인스턴스화된 UD_ENGLISH 말뭉치들은 일반 POS tags('POS'), 범용 POS tags('upos'), 형태학적 tags('tense', 'number'...) 등 여러 레이어의 주석을 가지고 있습니다. 다음 코드는 `label_type='upos'`를 인자로 사용하는 예시입니다.
```python
# 범용 POS tag 작업에 대한 레이블 사전을 만듭니다.
upos_dictionary = corpus.make_label_dictionary(label_type='upos')

# 사전을 출력합니다.
print(upos_dictionary)
```
결과입니다 :
```console
Dictionary with 17 tags: PROPN, PUNCT, ADJ, NOUN, VERB, DET, ADP, AUX, PRON, PART, SCONJ, NUM, ADV, CCONJ, X, INTJ, SYM
```

#### 다른 레이블 유형에 대한 사전
위의 예에서 `make_label_dictionary`를 호출하면 동일한 말뭉치에 있는 모든 레이블 유형에 대한 통계가 인쇄됩니다.
```console
Corpus contains the labels: upos (#204585), lemma (#204584), pos (#204584), dependency (#204584), number (#68023), verbform (#35412), prontype (#33584), person (#21187), tense (#20238), mood (#16547), degree (#13649), definite (#13300), case (#12091), numtype (#4266), gender (#4038), poss (#3039), voice (#1205), typo (#332), abbr (#126), reflex (#100), style (#33), foreign (#18)
```
UD_ENGLISH 말뭉치는 이런 레이블을 가지고 있으며 이에 대한 사전을 만들 수 있습니다. 아래의 예시는 일반 POS tags와 형태학적 숫자 tags에 관한 사전을 만드는 예시입니다.
```python
# 일반 POS tags를 위한 사전을 만듭니다.
pos_dictionary = corpus.make_label_dictionary(label_type='pos')

# 형태학적 숫자 tags를 위한 사전을 만듭니다.
tense_dictionary = corpus.make_label_dictionary(label_type='number')
```
만약 위 사전을 출력한다면 POS 사전에는 50개의 태그가 있고 이 말뭉치에 대한 숫자 사전이 2개(단수 및 복수)만 포함되어 있습니다.

#### 다른 말뭉치를 위한 사전
`make_label_dictionary` 메소드는 텍스트 분류 말뭉치를 포함하여 모든 말뭉치에 사용할 수 있습니다 :
```python
# 텍스트 분류 작업을 위해 레이블 사전을 만듭니다.
corpus = flair.datasets.TREC_6()
print(corpus.make_label_dictionary('question_class'))
```

**The MultiCorpus Object**부터 다시 하면 됨.
