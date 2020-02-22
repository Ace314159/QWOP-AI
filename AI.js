var G;
var agent;
var actions = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
];
var interval;

var prevScore = 0;
var prevAction = 0;
let prevState = [0, 0, 0, 0];

let prevGameTime;

function initAI(game) {
    G = game;
    G.onmousedown();

    prevState = getState();
    agent = new DQNAgent(getState().length);

    prevGameTime = (new Date()).getTime();
    setLearnInterval();
}

function setLearnInterval() {
    interval = setInterval(learn, 1000.0/10);
}

function getState() {
    return [
        G.rightHip.getJointAngle(),
        G.leftHip.getJointAngle(),
        //G.rightShoulder.getJointAngle(),
        //G.leftShoulder.getJointAngle(),
        //G.rightElbow.getJointAngle(),
        //G.leftElbow.getJointAngle(),
        G.rightKnee.getJointAngle(),
        G.leftKnee.getJointAngle(),
        //G.neck.getJointAngle(),
        //G.neck.getBodyA().getPosition().y,
    ];
}

function performAction(action) {
    const score = G.score;
    reward = score - prevScore;
    prevScore = score;
    prevAction = action;
    let state = getState();

    G.QDown = actions[action][0] == 1;
    G.WDown = actions[action][1] == 1;
    G.ODown = actions[action][2] == 1;
    G.PDown = actions[action][3] == 1;

    return {
        state,
        reward,
        done: G.gameEnded,
    }
}


async function learn() {
    if(interval === undefined) {
        return;
    }
    const action = await agent.act(getState());
    let {state, reward, done} = performAction(action);
    //reward -= G.neck.getBodyA().getPosition().y
    if(done || (new Date()).getTime() - prevGameTime >= 60 * 1000) {
        if(done) {
            reward = -10;
        }
        console.log("Scored", G.score);
        clearInterval(interval);
        interval = undefined;
        await agent.replay(32);
        G.reset();
        prevGameTime = (new Date()).getTime();
        setLearnInterval();
    }
    agent.memorize({state: prevState, action: prevAction, reward, nextState: state, done});
    prevState = state;
}
