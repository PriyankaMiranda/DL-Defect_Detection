import os
import time 
import numpy as np
import pickle

class Train_Class:

	def myprint(s):
		with open('modelresults.txt','w+') as f:
			print(s, file=f)

	def train(self, model):
		# Hyper parameters
		evaluate_every = 10 # interval for evaluating on one-shot tasks
		batch_size = 10
		val_batch_size = 20
		n_iter = 10000 # No. of training iterations
		N_way = 20 # how many classes for testing one-shot tasks
		n_val = 250 # how many one-shot tasks to validate on
		best = -1
		
		model_path = './weights/'
		save_path = 'assets/'

		with open(os.path.join(save_path, "train.pickle"), "rb") as f:
			(Xtrain, train_classes) = pickle.load(f)

		Xtrain = np.array(Xtrain)
		n_sets, w, h , channels	= Xtrain.shape
		train_classes = np.array(train_classes)

		with open(os.path.join(save_path, "train.pickle"), "rb") as f:
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

			if x % evaluate_every == 0:
				print("\n ------------- \n")
				print("Time for {0} iterations: {1} mins. Train Loss: {2}".format(x, (time.time()-t_start)/60.0), loss)
				loss_per_iter = "Loss: "+str(loss)+" Iteration: "+str(x)
				myprint(loss_per_iter)

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
				n_correct = 0

				for i in range(val_batch_size):
					if (curr_prob[i] >= 0.5):
						curr_prob[i] = 1
					else:
						curr_prob[i] = 0				
					if (curr_prob[i] == targets[i]):
						n_correct+=1

				percent_correct = (100.0 * n_correct / val_batch_size)
				print("Got an average of {0}% accuracy for {1} validation set  \n".format(percent_correct,val_batch_size))
				accu_per_iter = "Accu: "+str(loss)+" Iteration: "+str(x)+" Validation dataset size: "+str(val_batch_size)
				myprint(accu_per_iter)
				if percent_correct >= best:
					model.save_weights(os.path.join(model_path, 'my_model_weights.{}.h5'.format(x)))
					print("Current best: {0}, previous best: {1}".format(percent_correct, best))
					best_results = "Current best: "+str(percent_correct)+" previous best: "+str(best)
					myprint(best_results)
					best = percent_correct
		model.load_weights(os.path.join(model_path, "my_model_weights.final.h5"))








