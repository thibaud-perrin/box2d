import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Actor Model
        self.actor_model = self.build_actor_model()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Critic Model
        self.critic_model = self.build_critic_model()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        # Target networks
        self.target_actor = self.build_actor_model()
        self.target_actor.set_weights(self.actor_model.get_weights())

        self.target_critic = self.build_critic_model()
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Discount factor
        self.gamma = 0.99

        # Target update rate
        self.tau = 0.005

    def build_actor_model(self):
        model = tf.keras.Sequential([
            layers.Input((self.state_dim,)),
            layers.Dense(400, activation='relu'),
            layers.Dense(300, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh')
        ])
        return model

    def build_critic_model(self):
        # State input
        state_input = layers.Input((self.state_dim,))
        state_out = layers.Dense(16, activation='relu')(state_input)
        state_out = layers.Dense(32, activation='relu')(state_out)

        # Action input
        action_input = layers.Input((self.action_dim,))
        action_out = layers.Dense(32, activation='relu')(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(400, activation='relu')(concat)
        out = layers.Dense(300, activation='relu')(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Convert batch to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(np.reshape(next_state_batch, [1, self.state_dim]), training=True)
            y = reward_batch + self.gamma * self.target_critic([np.reshape(next_state_batch, [1, self.state_dim]), target_actions], training=True)
            critic_value = self.critic_model([np.reshape(state_batch, [1, self.state_dim]), np.reshape(action_batch, [1, self.action_dim])], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(np.reshape(state_batch, [1, self.state_dim]), training=True)
            critic_value = self.critic_model([np.reshape(state_batch, [1, self.state_dim]), actions], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        self.update_target(self.actor_model, self.target_actor)
        self.update_target(self.critic_model, self.target_critic)

    def update_target(self, model, target_model):
        model_weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]

        target_model.set_weights(target_weights)

    def get_action(self, state):
        return self.actor_model.predict(state)


# Initialize gym environment and the agent
env = gym.make('BipedalWalker-v3')
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])

# Iterate the game
for episode in range(1000):
    # Reset state in the beginning of each game
    state, _ = env.reset()
    episode_reward = 0

    while True:
        # Decide action
        action = agent.get_action(np.reshape(state, [1, agent.state_dim]))
        action = np.squeeze(action, axis=0)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Perform the action and get the next_state, reward, and done information
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Train the agent with the experience
        agent.update(state, action, reward, next_state)

        state = next_state
        if done:
            break

    print(f"Episode: {episode}, Reward: {episode_reward}")
