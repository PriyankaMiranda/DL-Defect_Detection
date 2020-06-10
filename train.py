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
		evaluate_every = 1 # interval for evaluating on one-shot tasks
		batch_size = 10
		val_batch_size = 4
		n_iter = 10000 # No. of training iterations
		N_way = 20 # how many classes for testing one-shot tasks
		n_val = 250 # how many one-shot tasks to validate on
		best = -1
		n_correct = 0

		model_path = './weights/'
		save_path = 'assets/'

		with open(os.path.join(save_path, "train.pickle"), "rb") as f:
			(Xtrain, train_classes) = pickle.load(f)

		Xtrain = np.array(Xtrain)
		n_sets, w, h , channels	= Xtrain.shape
		train_classes = np.array(train_classes)

		with open(os.path.join(save_path, "val.pickle"), "rb") as f:
			(Xval, val_classes) = pickle.load(f)

		Xval = np.array(Xval)
		n_sets_val, w_val, h_val , channels_val	= Xval.shape
		val_classes = np.array(val_classes)

		print("Starting training process!")
		print("-------------------------------------")
		t_start = time.time()

		for x in range(1, n_iter+1):
			y = []
			pairs=[np.zeros((batch_size, h, w, channels)) for i in range(2)]
			random_choices = np.random.choice(n_sets,size=(batch_size,),replace=False)
			targets=np.zeros((batch_size,))

			# 1st half has diff class(0's), 2nd half of batch has same class(1's)
			targets[batch_size//2:] = 1
			for i in range(batch_size):
				random_choice = random_choices[i]
				pairs[0][i,:,:,:] = Xtrain[random_choice].reshape(w, h, 3)
				# pick images of same class for 1st half, different for 2nd
				if i >= batch_size // 2:
					# find another random integer with the same class
					random_choice_2 = np.random.choice(n_sets,size=(1,),replace=False)
					while(train_classes[random_choice_2] != train_classes[random_choice]):
						random_choice_2 = np.random.choice(n_sets,size=(1,),replace=False)
				else: 
					# find a category with diff class
					random_choice_2 = np.random.choice(n_sets,size=(1,),replace=False)
					while(train_classes[random_choice_2] == train_classes[random_choice]):
						random_choice_2 = np.random.choice(n_sets,size=(1,),replace=False)
				pairs[1][i,:,:,:] = Xtrain[random_choice_2].reshape(w, h, 3)
			
			loss = model.train_on_batch(pairs,targets)
			print("Loss: "+str(loss)+" Iteration: "+str(x))

			if x % evaluate_every == 0:
				print("\n ------------- \n")
				print("Time for {0} iterations: {1} mins".format(x, (time.time()-t_start)/60.0))
				print("Train Loss: {0}".format(loss)) 
				
				pairs=[np.zeros((val_batch_size, h, w, channels)) for i in range(2)]
				random_choices = np.random.choice(n_sets_val,size=(val_batch_size,),replace=False)
				targets=np.zeros((val_batch_size,))
				targets[val_batch_size//2:] = 1

				for i in range(val_batch_size):
					random_choice = random_choices[i]
					pairs[0][i,:,:,:] = Xval[random_choice].reshape(w, h, 3)
					# pick images of same class for 1st half, different for 2nd
					if i >= val_batch_size // 2:
					# find another random integer with the same class
						random_choice_2 = np.random.choice(n_sets_val,size=(1,),replace=False)
						while(val_classes[random_choice_2] != val_classes[random_choice]):
							random_choice_2 = np.random.choice(n_sets_val,size=(1,),replace=False)
					else: 
					# find a category with diff class
						random_choice_2 = np.random.choice(n_sets_val,size=(1,),replace=False)
						while(val_classes[random_choice_2] == val_classes[random_choice]):
							random_choice_2 = np.random.choice(n_sets_val,size=(1,),replace=False)
				pairs[1][i,:,:,:] = Xval[random_choice_2].reshape(w, h, 3)

				curr_prob = model.predict(pairs)
				for i in range(val_batch_size):
					if (curr_prob[i] == targets[i]):
						n_correct+=1

				percent_correct = (100.0 * n_correct / val_batch_size)
				print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,len(val_batch_size)))
				model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(x)))
				if percent_correct >= best:
					print("Current best: {0}, previous best: {1}".format(percent_correct, best))
					best = percent_correct
		model.load_weights(os.path.join(model_path, "weights.final.h5"))








