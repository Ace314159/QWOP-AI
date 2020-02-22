class DQNAgent {
    constructor(numInputs, gamma=0.95, epsilon=1, epsilonMin=0.01, epsilonDecay=0.99, learningRate=0.001) {
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        this.learningRate = learningRate;
        this.model = getModel(numInputs);
        this.numInputs = numInputs
        this.model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse'],
        });
        this.memory = [];
        this.memoryI = 0;
        this.maxMemorySize = 2000;
        this.actionSize = 4;
    }

    memorize(m) {
        if(this.memory.length === this.maxMemorySize) {
            this.memory[this.memoryI] = m;
            this.memoryI = (this.memoryI + 1) % this.maxMemorySize;
        } else {
            this.memory.push(m);
        }
    }

    async act(state) {
        if(Math.random() <= this.epsilon) {
            return Math.floor(Math.random() * this.actionSize);
        }
        console.log("Using Model");
        let pred = this.model.predict(tf.tensor2d(state, [1, this.numInputs]));
        return (await pred.data()).reduce((acc, cur, idx, src) => cur > src[acc] ? idx : acc, 0);
    }

    async replay(batchSize) {
        batchSize = _.min([batchSize, this.memory.length]);
        let minibatch = _.take(_.shuffle(this.memory), batchSize);
        let states = [];
        let targetFs = [];
        for(let i = 0; i < batchSize; i++) {
            let {state, action, reward, nextState, done} = minibatch[i];
            let target = reward;
            if(!done) {
                let pred = this.model.predict(tf.tensor2d(nextState, [1, this.numInputs]));
                target = (reward + this.gamma * _.max(await pred.data()));
            }
            let targetF = this.model.predict(tf.tensor2d(state, [1, this.numInputs]));
            targetF = await targetF.data();
            targetF[action] = target;
            states.push(state);
            targetFs.push(targetF);
        }
        await this.model.fit(tf.tensor2d(states, [batchSize, this.numInputs]), tf.tensor2d(targetFs, [batchSize, this.actionSize]), {
            batchSize,
            epochs: 1,
            shuffle: true,
        });
        if(this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }
}