#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Define the gridworld size
const int GRID_SIZE = 5;

// Define the Q-learning agent class
class QLearning {
public:
    QLearning(float alpha, float gamma, float epsilon) :
        alpha(alpha), gamma(gamma), epsilon(epsilon), rng(std::random_device{}()) {
        // Initialize the Q-table with all values set to zero
        qTable.resize(GRID_SIZE, std::vector<float>(GRID_SIZE, 0.0f));
    }

    // Choose the best action for a given state
    int getBestAction(int state) const {
        int bestAction = 0;
        float bestValue = qTable[state / GRID_SIZE][state % GRID_SIZE];
        for (int action = 1; action < 4; ++action) {
            int nextState = getNextState(state, action);
            float value = qTable[nextState / GRID_SIZE][nextState % GRID_SIZE];
            if (value > bestValue) {
                bestAction = action;
                bestValue = value;
            }
        }
        return bestAction;
    }

    // Choose an action using epsilon-greedy exploration
    int chooseAction(int state) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < epsilon) {
            // Choose a random action with probability epsilon
            return std::uniform_int_distribution<int>(0, 3)(rng);
        }
        else {
            // Choose the best action for the current state with probability 1 - epsilon
            return getBestAction(state);
        }
    }

    // Update the Q-table based on the observed reward and next state
    void updateQTable(int state, int action, float reward, int nextState) {
        qTable[state / GRID_SIZE][state % GRID_SIZE] +=
            alpha * (reward + gamma * getMaxQValue(nextState) - qTable[state / GRID_SIZE][state % GRID_SIZE]);
    }

    // Decay the exploration rate
    void decayExplorationRate() {
        epsilon *= 0.99f;
    }

    // Print the Q-table
    void printQTable() const {
        std::cout << "Q-Table:\n";
        for (int i = 0; i < GRID_SIZE; ++i) {
            for (int j = 0; j < GRID_SIZE; ++j) {
                std::cout << qTable[i][j] << " ";
            }
            std::cout << "\n";
        }
    }

private:
    // Q-table
    std::vector<std::vector<float>> qTable;

    // Learning rate
    float alpha;

    // Discount factor
    float gamma;

    // Exploration rate
    float epsilon;

    // Random number generator
    std::mt19937 rng;

    // Get the next state given the current state and action
    int getNextState(int state, int action) const {
        int row = state / GRID_SIZE;
        int col = state % GRID_SIZE;

        switch (action) {
            case 0: // Up
                row = std::max(row - 1, 0);
                break;
            case 1: // Down
                row = std::min(row + 1, GRID_SIZE - 1);
                break;
            case 2: // Left
                col = std::max(col - 1, 0);
                break;
            case 3: // Right
                col = std::min(col + 1, GRID_SIZE - 1);
                break;
        }

        return row * GRID_SIZE + col;
    }

    // Get the maximum Q-value for a given state
    float getMaxQValue(int state) const {
        float maxQValue = qTable[state / GRID_SIZE][state % GRID_SIZE];
        for (int action = 1; action < 4; ++action) {
            int nextState = getNextState(state, action);
            float qValue = qTable[nextState / GRID_SIZE][nextState % GRID_SIZE];
            if (qValue > maxQValue) {
                maxQValue = qValue;
            }
        }
        return maxQValue;
    }
};

int main() {
    // Create an instance of the Q-learning agent
    QLearning agent(0.5f, 0.9f, 0.1f);

    // Run the Q-learning loop for a specified number of episodes
    const int numEpisodes = 100;
    for (int episode = 0; episode < numEpisodes; ++episode) {
        // Start in a random state
        int state = std::uniform_int_distribution<int>(0, GRID_SIZE * GRID_SIZE - 1)(agent.rng);

        // Perform actions until reaching the terminal state
        while (state != GRID_SIZE * GRID_SIZE - 1) {
            // Choose an action
            int action = agent.chooseAction(state);

            // Get the next state and reward
            int nextState = agent.getNextState(state, action);
            float reward = (nextState == GRID_SIZE * GRID_SIZE - 1) ? 1.0f : 0.0f;

            // Update the Q-table
            agent.updateQTable(state, action, reward, nextState);

            // Transition to the next state
            state = nextState;
        }

        // Decay the exploration rate after each episode
        agent.decayExplorationRate();
    }

    // Print the learned Q-table
    agent.printQTable();

    return 0;
}
