//Lets say I have a function softmax_words that outputs a matrix of the one hot vectors for top b words according to probabilities
b=5 // beam width
//Let sequences be a matrix of one hot vectors representing the current b sequences in consideration
for i in sequences:
  word_vectors=softmax_words(i)
  for j in word_vectors:
    k=i.copy()
    k.append(j)
    sequences.append(k)
  sequences.remove(i)
//now sequences has b*b number of sequences
//let eval_prob be a function to evaluate probability of a sequence
sequences=sorted(sequences,key=eval_prob())
sequences=sequences[:5]
  
