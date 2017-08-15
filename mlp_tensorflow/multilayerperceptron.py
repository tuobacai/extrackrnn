#get the mnist data
# wget http://deeplearning.net/data/mnist/mnist.pkl.gz
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
class MultilayerPerceptron(object):
    def __init__(self):
        self.learning_rate=0.001
        self.training_epochs=30
        self.batch_size=100
        self.display_step=1

        self.n_hidden_1=256
        self.n_hidden_2=512
        self.n_input=784
        self.n_classes=10

    def load(self):
        self.mnist = input_data.read_data_sets("./mnist/", one_hot=True)

    def create_model(self,x,weights,biases):
        layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
        layer_1=tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

        layer_3=tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
        layer_3=tf.nn.relu(layer_3)

        out_layer=tf.add(tf.matmul(layer_3,weights['out']),biases['out'])
        return out_layer

    def run(self):
        # tf Graph input
        x = tf.placeholder("float", [None, self.n_input])
        y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & biases
        weights = {
            # you can change
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'h3':tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'b3':tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        pred = self.create_model(x, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.training_epochs):
                avg_cost=0
                total_batch=int(self.mnist.train.num_examples/self.batch_size)
                for i in range(total_batch):
                    batch_x,batch_y=self.mnist.train.next_batch(self.batch_size)
                    _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                    avg_cost+=c/total_batch

                if epoch%self.display_step==0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=","{:.9f}".format(avg_cost))
            print ("Optimization Finished!")

            corrext_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
            accuracy=tf.reduce_mean(tf.cast(corrext_prediction,'float'))
            print("Accuracy:", accuracy.eval({x:self.mnist.test.images, y: self.mnist.test.labels}))
if __name__=='__main__':
    mlp=MultilayerPerceptron()
    mlp.load()
    mlp.run()