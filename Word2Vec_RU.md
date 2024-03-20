# Общее описание и реализация Word2Vec с помощью PyTorch

<img src="https://community.alteryx.com/t5/image/serverpage/image-id/45458iDEB69E518EBA3AD9?v=v2" alt="Word2Vec" height="400">

## Аннотация

В данной статье даётся общее описание векторного представления вложений слов - модель `word2vec`. Также рассматривается пример реализации модели `word2vec` с использованием библиотеки `PyTorch`. Приведена реализация как архитектуры `slip-gram` так и `CBOW`.

[Исходный код](https://github.com/AntonSHBK/word2vec_pytorch_realization)

`Word2Vec` — это популярная модель обучения вложений слов, предложенная исследователями `Google` в 2013 году (Томас Миколов). Она позволяет преобразовать слова из корпуса текстов в векторы чисел таким образом, что слова с похожими семантическими значениями имеют близкие векторные представления в многомерном пространстве. Это делает `Word2Vec` мощным инструментом для задач обработки естественного языка (`NLP`), таких как анализ тональности, машинный перевод, автоматическое резюмирование и многие другие.

Основные характеристики `Word2Vec`:
* Распределенное представление: Каждое слово представляется вектором в многомерном пространстве, где отношения между словами отражаются через косинусное сходство между их векторами.
* Обучение без учителя: Word2Vec обучается на больших неразмеченных текстовых корпусах без необходимости во внешних аннотациях или разметке.
* Контекстное обучение: Векторы слов получаются на основе контекста, в котором эти слова встречаются, что позволяет захватить их семантические и синтаксические отношения.
## Две основные архитектуры модели Word2Vec:
`CBOW` (Continuous Bag of Words): Этот подход предсказывает текущее слово на основе контекста вокруг него. Например, для фразы "синее небо над головой", модель `CBOW` будет пытаться предсказать слово "небо" на основе контекстных слов "синее", "над", "головой". `CBOW` быстро обрабатывает большие объемы данных, но менее эффективен для редких слов.

`Skip-Gram`: В этом подходе наоборот, используется текущее слово для предсказания слов в его контексте. Для того же примера, модель `Skip-Gram` будет пытаться предсказать слова "синее", "над", "головой" на основе слова "небо". `Skip-Gram` медленнее обрабатывает данные, но лучше работает с редкими словами и менее частыми контекстами.

## CBOW (Continuous Bag of Words)
Целью `CBOW` является предсказание целевого слова на основе контекста вокруг этого слова. Контекст определяется как набор слов вокруг целевого слова в пределах заданного окна. Архитектура модели упрощенно представляет собой трехслойную нейронную сеть: входной слой, скрытый слой и выходной слой.

<!-- <img src="https://www.researchgate.net/profile/Raouf-Ganda/publication/318975052/figure/fig2/AS:631670868820002@1527613479312/CBOW-architecture-predicts-the-current-word-based-on-the-context.png" alt="CBOW" height="600"> -->


Входной слой: На вход модели подаются контекстные слова. Эти слова представляются в виде векторов с использованием "one-hot encoding", где каждый вектор имеет размерность, равную размеру словаря, и содержит 1 на позиции, соответствующей индексу слова в словаре, и 0 в остальных позициях.

Скрытый слой: Векторы входных слов умножаются на матрицу весов между входным и скрытым слоем, результатом чего является вектор скрытого слоя. Для CBOW вектора контекстных слов обычно усредняются перед передачей на следующий слой.

Выходной слой: Вектор скрытого слоя умножается на матрицу весов между скрытым и выходным слоем, результат чего проходит через `softmax`-функцию для получения вероятностей каждого слова в словаре быть целевым словом. Цель обучения - максимизировать вероятность правильного целевого слова.

### Skip-Gram
В отличие от CBOW, цель Skip-Gram - предсказать контекстные слова для данного целевого слова. Это слово на входе модели используется для предсказания слов в его контексте в пределах заданного диапазона слов (называют окном).

<!-- <img src="https://www.researchgate.net/profile/Firas-Odeh/publication/327537608/figure/fig5/AS:668724143063056@1536447668875/word2vec-Skip-gram-model-Image-credit-User-Moucrowap-on-Wikipedia.ppm" alt="Skip-Gram" height="600"> -->

Входной слой: Входом является целевое слово, представленное вектором `one-hot`.
Скрытый слой: Такой же, как и в `CBOW`, где вектор целевого слова умножается на матрицу весов, ведущую к скрытому слою.

Выходной слой: В отличие от `CBOW`, где выходной слой представляет собой один `softmax`, в `Skip-Gram` для каждого слова в контексте используется отдельный `softmax`, что означает, что модель пытается предсказать каждое контекстное слово отдельно. Цель обучения - максимизировать вероятность появления реальных контекстных слов для данного целевого слова.

## Реализация Pytorch
Реализуем модель Word2Vec с архитектурой `Skip-Gram` с использованием библиотеки `PyTorch`. Это не самая лучшая реализация Word2Vec, но мой взгляд достаточно простая.

Импортируем все необходимые библиотеки:
```py
import re
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
```

### Skip-Gram: 
__Подготовка данных__

Для начала нам нужно подготовить наши данные. В этом примере мы опустим этап предобработки и сосредоточимся на самой модели. Предположим, у нас уже есть словарь соответствия слов к индексам и наоборот, а также данные для обучения в формате пар (центральное слово, контекстное слово).

```py
def prepare_data(text, window_size=2):
	# Удаляем все символы кроме a-z, @, и #
	text = re.sub(r'[^a-z@# ]', '', text)    
	# Преобразуем текст в нижний регистр
	text = text.lower()
	# Разбиваем по словам
	tokens = text.split()    
	# Формируем словарь уникальных слов
	vocab = set(tokens)
	# Формируем слова слов с указанием индекса  слова в словаре
	word_to_ix = {word: i for i, word in enumerate(vocab)}
	# Формируем пары слов n-грамм
	data = []
	for i in range(len(tokens)):
		for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
			if i != j:
				data.append((tokens[i], tokens[j]))    
	return data, word_to_ix, len(vocab)
```

__Определение модели Skip-Gram__

Установим структуру датасета для даталоадера, можно обойтись и без этого, однако это позволяет масштабировать проект и возможно пригодится нам в дальнейшем. Подробно можно ознакомиться в [официальной документации](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

Для класса Dataset требуются следующие три метода:
* `__init__`: выполняется при создании экземпляра класса. Обычно здесь определяются атрибуты.
* `__len__`: должно возвращать длину набора данных. Это важно для понимания того, сколько памяти выделить.
* `__getitem__`: учитывая индекс, возражает данные в виде батча (набор данных установленной длины), соответствующий этому индексу.

```py
class SkipGramModelDataset(Dataset):
	def __init__(self, data, word_to_ix):
		self.data = [(word_to_ix[center], word_to_ix[context]) for center, context in data]	
	def __len__(self):
		return len(self.data)	
	def __getitem__(self, idx):
		return torch.tensor(self.data[idx][0], dtype=torch.long), torch.tensor(self.data[idx][1], dtype=torch.long)
		
```	

Определим структуру нашей модели на `PyTorch`. 

Структуру модели примем простой, входной слой - `nn.Embedding` стандартный для задач `NLP`, представляет собой векторное представление (вложений) слов. Далее идёт линейный слой. В завершении используем логарифмированную функцию `softmax`.

`LogSoftmax` обычно применяется к последнему слою нейронной сети перед вычислением функции потерь, например, `NLLLoss` (Negative Log Likelihood Loss). `LogSoftmax` преобразует логиты (выходы линейного слоя) в логарифмированные вероятности, которые затем можно напрямую использовать с `NLLLoss`. Важно, что `NLLLoss` ожидает, что входные данные для неё будут в формате логарифмированных вероятностей.

```py
class Word2VecSkipGramModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim):
		super(Word2VecSkipGramModel, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.out_layer = nn.Linear(embedding_dim, vocab_size)
		self.activation_function = nn.LogSoftmax(dim=-1)

	def forward(self, center_word_idx):
		hidden_layer = self.embeddings(center_word_idx)
		out_layer = self.out_layer(hidden_layer)
		log_probs = self.activation_function(out_layer)
		return log_probs
```
__Обучение модели__

Общий подход:
1. Инициализация- сначала векторы слов инициализируются случайными значениями.
2. Прогнозирование контекста - для каждого слова в обучающем корпусе модель использует его векторное представление (выход первого слоя нейронной сети) для предсказания векторов слов в его контексте (через выход второго слоя и функцию `softmax`).
3. Оптимизация - функция потерь оптимизируется для улучшения предсказаний контекста. Это обновляет вектора слов в процессе обучения.
4. Итерации - процесс повторяется на протяжении нескольких эпох обучения.
```py
def train_model(data, word_to_ix, vocab_size, embedding_dim=50, epochs=10, batch_size=1):
	# Формируем набор данных
	dataset = SkipGramModelDataset(data, word_to_ix)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	# модель
	model = Word2VecSkipGramModel(vocab_size, embedding_dim)
	# функция потерь
	loss_function = nn.NLLLoss()
	#  оптимизатор
	optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

	for epoch in range(epochs):
		total_loss = 0
		for center_word, context_word in dataloader:
			model.zero_grad()
			log_probs = model(center_word)
			loss = loss_function(log_probs, context_word)
			loss.backward()
			optimizer.step()            
			total_loss += loss.item()			
		print(f'Epoch {epoch + 1}, Loss: {total_loss}')
	return model
```

```py
# Основная функция для вызова
def train(data: str):
	# Гиперпараметры:
	# размер окна
	window_size = 2
	# длина ембединга
	embedding_dim = 10
	# количество эпох обучения
	epochs = 5
	# размер батча
	batch_size = 1
	
	# предобработка данных
	ngramm_data, word_to_ix, vocab_size = prepare_data(data, window_size) 
	# основной процесс формирование и обучения модели
	model = train_model(ngramm_data, word_to_ix, vocab_size, embedding_dim, epochs, batch_size)
	
	# # Извлекаем векторы слов из модели
	embeddings = model.embeddings.weight.data.numpy()
	# формируем словарь слов и их векторное представление
	ix_to_word = {i: word for word, i in word_to_ix.items()}
	w2v_dict = {ix_to_word[ix]: embeddings[ix] for ix in range(vocab_size)}
	return w2v_dict
```
Гиперпараметры выбраны исключительно в учебных целях.
```py
# Тестовые данные
test_text = 'Captures Semantic Relationships: The skip-gram model effectively captures semantic relationships between words. It learns word embeddings that encode similar meanings and associations, allowing for tasks like word analogies and similarity calculations. Handles Rare Words: The skip-gram model performs well even with rare words or words with limited occurrences in the training data. It can generate meaningful representations for such words by leveraging the context in which they appear. Contextual Flexibility: The skip-gram model allows for flexible context definitions by using a window around each target word. This flexibility captures local and global word associations, resulting in richer semantic representations. Scalability: The skip-gram model can be trained efficiently on large-scale datasets due to its simplicity and parallelization potential. It can process vast amounts of text data to generate high-quality word embeddings.'

w2v_dict = train(test_text)
```
Мы создаем датасет с центральными словами и их контекстами, правильно формируем входы и цели для обучения.
Модель `Word2VecSkipGramModel` принимает индекс центрального слова, возвращает логарифмированные вероятности для всех слов в словаре.
В функции обучения `train_model` используем `NLLLoss` для вычисления потерь между предсказанными логарифмированными вероятностями.

### CBOW:
В этом подходе, модель предсказывает текущее слово на основе контекста вокруг него. Это значит, что на вход модели подаются несколько слов из контекста текущего слова, и модель учится предсказывать это текущее слово.

__Основные изменения__

1. Подготовка данных
   
Меняем функцию подготовки данных так, чтобы она создавала обучающие примеры, состоящие из контекстных слов в качестве входных данных и центрального слова как цели:

```py
def prepare_data_cbow(text: str, window_size=2):
	text = re.sub(r'[^a-z@# ]', '', text.lower())    
	tokens = text.split()    
	vocab = set(tokens)
	word_to_ix = {word: i for i, word in enumerate(vocab)}
	data = []
	for i in range(window_size, len(tokens) - window_size):
		context = [tokens[i - j - 1] for j in range(window_size)] + [tokens[i + j + 1] for j in range(window_size)]
		target = tokens[i]
		data.append((context, target))
	return data, word_to_ix, len(vocab)	

class SkipGramDataset(Dataset):
	def __init__(self, data, word_to_ix):			
		self.data = [(word_to_ix[center], word_to_ix[context]) for center, context in data]
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return torch.tensor(self.data[idx][0], dtype=torch.long), torch.tensor(self.data[idx][1], dtype=torch.long)
```
2. Изменение архитектуры модели

Модифицируйте модель так, чтобы она принимала контекстные слова и предсказывала центральное слово:
```py
class Word2VecCBOWModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim):
		super(Word2VecCBOWModel, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.out_layer = nn.Linear(embedding_dim, vocab_size)
		self.activation_function = nn.LogSoftmax(dim=1)

	def forward(self, center_word_idx):
		hidden_layer = torch.mean(self.embeddings(center_word_idx), dim=1)
		out_layer = self.out_layer(hidden_layer)
		log_probs = self.activation_function(out_layer)
		return log_probs
```
3. Обновление функции обучения
   
Нужно будет адаптировать функцию обучения для работы с новым форматом данных и моделью CBOW:
```py
def train_model_cbow(data, word_to_ix, vocab_size, embedding_dim=50, epochs=10, batch_size=1):
	dataset = CBOWDataset(data, word_to_ix)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	model = Word2VecCBOWModel(vocab_size, embedding_dim)
	loss_function = nn.NLLLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
	for epoch in range(epochs):
		total_loss = 0
		for context_words, target_word in dataloader:
			context_words = context_words
			model.zero_grad()
			log_probs = model(context_words)
			loss = loss_function(log_probs, target_word)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f'Epoch {epoch + 1}, Loss: {total_loss}')
	return model
```

## Улучшения производительности
Для повышения качества модели Word2Vec можно применить ряд методов и техник:
1. Увеличение объема тренировочных данных
    * Больше текстовых данных: Больший и более разнообразный тренировочный корпус может помочь модели лучше понять различные контексты использования слов и улучшить качество векторных представлений.
2. Предобработка данных
   * Токенизация: Эффективное разбиение текста на слова, предложения и другие значимые единицы.
   * Удаление стоп-слов: Исключение часто встречающихся слов, которые могут не нести значимой семантической нагрузки (например, предлоги, союзы).
   * Лемматизация и стемминг: Приведение слов к их базовой форме может помочь снизить размер словаря и уменьшить шум.
   * Использование n-грамм: Обучение модели на фразах или комбинациях слов (например, "Нью-Йорк" вместо "Нью" и "Йорк" отдельно) может улучшить качество вложений для составных терминов.
3. Настройка гиперпараметров
   * Размер вектора: Увеличение размера векторного представления слов может улучшить качество, захватывая больше нюансов семантики, но также увеличивает требования к вычислительным ресурсам и памяти.
   * Размер окна контекста: Экспериментирование с размером окна может помочь настроить баланс между изучением ближайшего контекста и более широкими контекстуальными отношениями.
   * Частота обновления (`subsampling`): Игнорирование чрезмерно часто встречающихся слов во время обучения может улучшить общее качество модели.
   * Количество эпох: Увеличение количества проходов по датасету может помочь модели лучше обучиться, но с риском переобучения.
4. `Negative Sampling` и `Hierarchical Softmax`
   * Количество негативных образцов: Настройка количества негативных образцов для каждого положительного образца может влиять на скорость и качество обучения.
   * Использование иерархического софтмакса: Может ускорить обучение для очень больших словарей за счет более эффективного расчета вероятностей.
5. Использование ансамблей и мультимодальных данных
   * Ансамбли моделей: Комбинирование предсказаний нескольких моделей может улучшить общее качество вложений.
   * Мультимодальное обучение: Интеграция информации из различных источников (текст, изображения, звук) может помочь создать более богатые и разнообразные представления слов.
6. Обратная связь и итеративное улучшение
   * Оценка качества: Регулярное тестирование модели на задачах, близких к целевому применению, поможет выявить слабые стороны и направления для улучшения.
   * Итеративное улучшение: Непрерывное добавление новых данных и переобучение модели с учетом полученной обратной связи может постоянно повышать ее качество.
   * Применение этих методов и техник требует экспериментирования и может варьироваться в зависимости от конкретных задач и доступных данных.

## Заключение:
`Word2Vec` предоставил прорыв в области `NLP`, предложив эффективный способ извлечения и представления семантических и синтаксических отношений между словами в виде векторов. Его вложения используются в различных задачах `NLP` и по сей день остаются важным инструментов для работы с текстовыми данными.

Используемые Источники:
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Основы Natural Language Processing для текста](https://habr.com/ru/companies/Voximplant/articles/446738/)
- [NLP — Преобразование текста: Word2Vec](https://habr.com/ru/companies/otus/articles/574624/)
- [Word2Vec: как работать с векторными представлениями слов](https://neurohive.io/ru/osnovy-data-science/word2vec-vektornye-predstavlenija-slov-dlja-mashinnogo-obuchenija/)
- [Word2Vec: покажи мне свой контекст, и я скажу, кто ты](https://sysblok.ru/knowhow/word2vec-pokazhi-mne-svoj-kontekst-i-ja-skazhu-kto-ty/)