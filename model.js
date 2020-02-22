function getModel(numInputs) {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 24,
        activation: 'relu',
        inputDim: numInputs,
        useBias: true,
    }));
    model.add(tf.layers.dense({
        units: 24,
        activation: 'relu',
        useBias: true,

    }));
    model.add(tf.layers.dense({
        units: 4,
        activation: 'linear',
        useBias: true,
    }));
    
    return model;
}