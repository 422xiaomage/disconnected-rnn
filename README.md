# disconnected-rnn
[Disconnected Recurrent Neural Networks for Text Categorization](http://aclweb.org/anthology/P18-1215), accepted by [ACL 2018](https://acl2018.org/paper/374/).<br />
This paper is produced by Baoxin Wang, but they may not open source their code. I implement it in keras and try to reproduce the results in the paper.<br />
The DRNN layer is written in the class DRNN. The input size is (batch_size, sequence_length, input_dim), output size is (batch_size, sequence_length, hidden_dim). It could be compared with Conv1d or GRU(return_sequences=True). Usually a GlobalMaxPooling1D layer is added after the DRNN layer. In the paper they used MLP and BatchNormalization.
