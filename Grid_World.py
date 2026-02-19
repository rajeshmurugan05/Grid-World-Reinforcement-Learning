import numpy as np

rows = 5
columns = 5
win_state = (4, 4)
jump_state = (3, 3)
initial_state = (1, 0)
black_box = [(3, 2), (2, 2), (2, 3), (2, 4)]
deterministic = False


class State:
    def __init__(self, state=initial_state):
        self.board = np.zeros([rows, columns])
        self.state = state
        self.isEnd = False
        self.determine = deterministic

    def giveReward(self, flag):
        if self.state == win_state and flag == "win":
            return 10
        elif self.state == win_state and flag == "jump":
            return 15
        else:
            return 0

    def isendfunction(self):
        if (self.state == win_state):
            self.isEnd = True

    def _select_actionProb(self, action):
        if action == "north":
            return np.random.choice(["north", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "south":
            return np.random.choice(["south", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "west":
            return np.random.choice(["west", "north", "south"], p=[0.8, 0.1, 0.1])
        if action == "east":
            return np.random.choice(["east", "north", "south"], p=[0.8, 0.1, 0.1])

    def next_position(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        if self.determine:
            if action == "north":
                next_state = (self.state[0] - 1, self.state[1])
            elif action == "south":
                next_state = (self.state[0] + 1, self.state[1])
            elif action == "west":
                next_state = (self.state[0], self.state[1] - 1)
            else:
                next_state = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            action = self._select_actionProb(action)
            self.determine = True
            next_state = self.next_position(action)

        if (next_state[0] >= 0) and (next_state[0] <= 4):
            if (next_state[1] >= 0) and (next_state[1] <= 4):
                if next_state not in black_box:
                    return next_state
                if next_state == (2, 3) and action == "south":
                    self.state = (3, 3)
        return self.state


class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["north", "south", "west", "east"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.2
        self.exp_rate = 0.3
        self.decay_gamma = 0.9

        # initial Q values
        self.Q_values = {}
        for i in range(rows):
            for j in range(columns):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = -1

    def select_action(self):
        mx_nxt_reward = 0
        action = ''

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def run_action(self, action):
        position = self.State.next_position(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def run(self, rounds=100):
        i = 0
        cum_itr_list = []
        cum_itr = False
        while (i < rounds and cum_itr == False):
            print("-----------------------------------------------> episode ", i)
            # to the end of game back propagate reward
            print(self.Q_values[self.State.state])
            if self.State.isEnd:
                # back propagate
                print("___________________ executed_________________")
                if [(1, 3), "south"] in (self.states):
                    print(self.states)
                    reward = self.State.giveReward("jump")
                else:
                    reward = self.State.giveReward("win")
                cum_itr_list.append(reward)
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    try:
                        current_q_value = self.Q_values[s[0]][s[1]]
                    except:
                        current_q_value = -1
                    print(current_q_value)
                    if s[0] == (1, 3) and s[1] == "south":
                        reward = 5
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    else:
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)

                if cum_itr_list[-30:].count(15) == 30:
                    print(cum_itr_list[-30:])
                    cum_itr = True
                    break

                self.reset()
                i += 1

            else:
                action = self.select_action()
                # append trace
                self.states.append([(self.State.state), action])
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.run_action(action)
                # mark is end
                self.State.isendfunction()
                print("nxt state", self.State.state)
                print("---------------------")
                self.isEnd = self.State.isEnd

    def give_results(self):
        for i in range(0, rows):
            print('----------------------------------')
            out = '| '
            for j in range(0, columns):
                out += str(max(self.Q_values[(i, j)].values())).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)

    ag.run(500)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    ag.give_results()
