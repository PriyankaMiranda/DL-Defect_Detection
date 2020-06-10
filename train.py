import os
import time 
import numpy as np
import pickle

class Train_Class:

	def make_oneshot_task(N, s="val", language=None):
		"""Create pairs of test image, support set for testing N way one-shot learning. """
		if s == 'train':
			X = Xtrain
			categories = train_classes
		else:
			X = Xval
			categories = val_classes
		n_classes, n_examples, w, h = X.shape

		indices = np.random.randint(0, n_examples,size=(N,))
		if language is not None: # if language is specified, select characters for that language
			low, high = categories[language]
			if N > high - low:
				raise ValueError("This language ({}) has less than {} letters".format(language, N))
			categories = np.random.choice(range(low,high),size=(N,),replace=False)
		else: # if no language specified just pick a bunch of random letters
			categories = np.random.choice(range(n_classes),size=(N,),replace=False)            
		true_category = categories[0]
		ex1, ex2 = np.random.choice(n_examples,replace=False,size=(2,))
		test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
		support_set = X[categories,indices,:,:]
		support_set[0,:,:] = X[true_category,ex2]
		support_set = support_set.reshape(N, w, h,1)
		targets = np.zeros((N,))
		targets[0] = 1
		targets, test_image, support_set = shuffle(targets, test_image, support_set)
		pairs = [test_image,support_set]

		return pairs, targets

	def test_oneshot(model, N, k, s = "val", verbose = 0):
		"""Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
		n_correct = 0
		if verbose:
			print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
		for i in range(k):
			inputs, targets = make_oneshot_task(N,s)
			probs = model.predict(inputs)
			if np.argmax(probs) == np.argmax(targets):
				n_correct+=1
		percent_correct = (100.0 * n_correct / k)
		if verbose:
			print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
		return percent_correct


	def train(self, model):

		# Hyper parameters
		evaluate_every = 200 # interval for evaluating on one-shot tasks
		batch_size = 10
		eval_batch_size = 10
		n_iter = 20000 # No. of training iterations
		N_way = 20 # how many classes for testing one-shot tasks
		n_val = 250 # how many one-shot tasks to validate on
		best = -1

		model_path = './weights/'
		save_path = 'assets/'

		with open(os.path.join(save_path, "train.pickle"), "rb") as f:
			(Xtrain, train_classes) = pickle.load(f)

		with open(os.path.join(save_path, "val.pickle"), "rb") as f:
			(Xval, val_classes) = pickle.load(f)

		print("Starting training process!")
		print("-------------------------------------")
		t_start = time.time()

		for i in range(1, n_iter+1):
			for x in range(batch_size):
				rand_pos = np.random.randint(len(train_classes)-1, size=(2, 1))
				X.append([Xtrain[rand_pos[0]],Xtrain[rand_pos[1]]])




				# if K.image_data_format() == 'channels_first':
				# 	x_train[rand_pos[0]] = x_train[rand_pos[0]].reshape(x_train.shape[0], 1, img_rows, img_cols)
				# 	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
				# 	print(x_train.shape)
				# 	input_shape = (1, img_rows, img_cols)
				# else:
				# 	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
				# 	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
				# 	input_shape = (img_rows, img_cols, 1)


				if (train_classes[rand_pos[0]] == train_classes[rand_pos[1]]):
					curr_y = 1
					y.append(1) # same classes
				else:
					curr_y = 0
					y.append(0) # diff classes

				loss = model.train_on_batch([Xtrain[rand_pos[0]],Xtrain[rand_pos[1]]], curr_y)
				input()

			if i % evaluate_every == 0:
				print("\n ------------- \n")
				print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
				print("Train Loss: {0}".format(loss)) 
				X_eval = []
				y_eval = []
				for x in range(eval_batch_size):
					rand_pos = np.random.randint(len(train_classes)-1, size=(2, 1))
					X_eval.append([Xval[rand_pos[0]],Xval[rand_pos[1]]])				
					if (val_classes[rand_pos[0]] == val_classes[rand_pos[1]]):
						curr_y = 1
						y_eval.append(1) # same classes
					else:
						curr_y = 0
						y_eval.append(0) # diff classes
					curr_prob = model.predict([Xval[rand_pos[0]][0],Xval[rand_pos[1]][0]])
					# prob.append(curr_prob)
					if curr_prob == curr_y:
						n_correct+=1

				# for x in range(len(prob)):
				# 	if np.argmax(prob[x]) == np.argmax(y[x]):

				percent_correct = (100.0 * n_correct / len(y_eval))
				print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,len(probs)))

				model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
				if percent_correct >= best:
					print("Current best: {0}, previous best: {1}".format(percent_correct, best))
					best = percent_correct

		model.load_weights(os.path.join(model_path, "weights.20000.h5"))








