import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer


# 1.3.1
def n1():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.array([0, 0, 0])
	y = np.array([1, 2, 2])
	w = np.array([3, 0, 0])
	z = np.array([3, 3, 3])
	ax.scatter(0, 0, 0)
	ax.scatter(1, 2, 2)
	ax.scatter(3, 0, 0)
	ax.scatter(3, 3, 3)

	print("Евклидово расстояние. x-y:", np.linalg.norm(x - y))
	print("Квадрат евклидова расстояния. y-w:", np.linalg.norm(y - w) ** 2)
	print("Расстояние Чебышева. w-z:", np.linalg.norm((w - z), ord=np.inf))
	print("Хеммингово расстояние. z-x:", np.linalg.norm((z - x), ord=1))
	plt.show()


# n1()


# 1.3.2
def n2():
	Z = np.zeros((5, 5))
	Z += np.arange(5)
	print(Z)


# n2()


# 2.3.1
def n3():
	iris = sns.load_dataset('iris')

	X_train, X_test, y_train, y_test = train_test_split(
		iris.iloc[:, :-1],
		iris.iloc[:, -1],
		test_size=0.15
	)
	X_train.shape, X_test.shape, y_train.shape, y_test.shape

	model = KNeighborsClassifier(n_neighbors=5)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	plt.figure(figsize=(10, 7))
	sns.scatterplot(x='petal_width', y='petal_length', data=iris, hue='species', s=70)
	plt.xlabel('Длина лепестка, см')
	plt.ylabel('Ширина лепестка, см')
	plt.legend(loc=2)
	plt.grid()
	# Перебираем все объекты из теста
	for i in range(len(y_test)):
		# Если предсказание неправильное
		if np.array(y_test)[i] != y_pred[i]:
			# то подсвечиваем точку красным
			plt.scatter(X_test.iloc[i, 3], X_test.iloc[i, 2], color='red', s=150)
	print(f'accuracy: {accuracy_score(y_test, y_pred) :.3}')

	plt.show()


n3()


# 3.3.2
def n4():
	data = [{"карий":1}, {"голубой":2}, {"карий":3}, {"голубой":4}, {"серый":5}, {"голубой":6}]

	dictvectorizer = DictVectorizer(sparse=False)
	features = dictvectorizer.fit_transform(data)
	print(features)


n4()