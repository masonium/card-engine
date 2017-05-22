use germanwhist::{self, Action, ActionError, GameEvent,
                  Round, PlayerState, ScoringRules};

use ndarray::prelude::*;
use rand::{Rng, thread_rng};

pub enum LearningModelError {
    MismatchedSize
}

/// Reinforcement Learning traits and implementations
pub trait LearningModel {
    //
    fn evaluate_q(&self, ArrayView<f32, Ix1>) -> f32;

    // Compute q, and the gradient.
    fn evaluate_q_grad(&self, p: &ArrayView<f32, Ix1>, grad: ArrayViewMut<f32, Ix1>) -> f32;

    // input size
    fn input_size(&self) -> usize;

    // num parameters
    fn num_parameters(&self) -> usize;

}

pub struct SarsaLambdaParameters {
    lambda: f32,
    gamma: f32,
    eps: f32
}

pub struct SarsaPlayer {
    state: PlayerState,
    e_trace: Array<f32, Ix1>,
    last_q: f32
}

impl SarsaPlayer {
    fn new(state: PlayerState, model_size: usize) -> SarsaPlayer {
        let e_trace = Array::zeros(model_size);

        SarsaPlayer { state,  e_trace, last_q: 0.0 }
    }

    /// Choose an epsilon-greedy action.
    fn epsilon_greedy_action<M: LearningModel>(&self, model: &M, eps: f32,
                                               actions: &[Action], mut sa: ArrayViewMut<f32, Ix1>) -> Action {
        /// epsilon-greedy state-choosing
        let mut rng = thread_rng();
        let r = rng.next_f32();

        // choose a random action with probability epsilon
        if r < eps {
            let random_action = rng.choose(actions).expect("must have positive number of actions");
            self.state.state_action_vector(sa.view_mut(), true, Some(random_action));
            *random_action
        } else {
            self.state.state_action_vector(sa.view_mut(), true, None);
            self.greedy_action(model, actions, sa)
        }
    }

    /// Fill in the state action input `sa` most beneficial action of those provided in the
    /// current state, returning the Q-value at that state.
    fn greedy_action<M: LearningModel>(&self, model: &M, actions: &[Action], mut sa: ArrayViewMut<f32, Ix1>) -> Action {
        let (max_action, _) = actions.iter().zip(actions.iter().map(|a| {
            self.state.state_action_vector(sa.view_mut(), false, Some(a));
            model.evaluate_q(sa.view())
        })).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).expect("action list should not be empty");

        self.state.state_action_vector(sa, false, Some(max_action));
        *max_action
    }

}

pub struct SarsaLambda<M: LearningModel> {
    players: [SarsaPlayer; 2],
    model: M,
    engine: germanwhist::Round,
    param: SarsaLambdaParameters
}

impl<M: LearningModel> SarsaLambda<M> {
    pub fn new(rules: ScoringRules, model: M, param: SarsaLambdaParameters) -> Result<SarsaLambda<M>, LearningModelError> {
        let players = [SarsaPlayer::new(PlayerState::new(0), model.num_parameters()),
                       SarsaPlayer::new(PlayerState::new(1), model.num_parameters())];

        if players[0].state.state_vector_size() != model.input_size() {
            return Err(LearningModelError::MismatchedSize);
        }

        Ok(SarsaLambda { players,
                         engine: Round::new(rules),
                         model,
                         param })
    }

    pub fn train_on_episode(&mut self) -> Result<(), ActionError> {
        // start a new round
        let ev = self.engine.start_round(None);
        for i in 0..2 {
            self.players[i].e_trace.fill(0.0);
            for e in &ev[i] {
                self.players[i].state.on_event(e)
            }
        }

        let mut player_action = Array::zeros(self.players[0].state.state_vector_size());
        let mut grad = Array::zeros(self.players[0].state.state_vector_size());

        // Evaluate the episode, tracking and updating the trace for each player
        while !self.engine.is_game_over() {
            // Get the next action based on whose turn it is.
            let active = self.engine.active_player();

            // choose the epsilon-greedy action for that player.
            let possible_actions = self.engine.possible_actions();
            let chosen_action = self.players[active].epsilon_greedy_action(
                &self.model, self.param.eps, &possible_actions, player_action.view_mut());

            // evaluate the gradient for the state-action pair
            let q_predict = self.model.evaluate_q_grad(&player_action.view(), grad.view_mut());

            // update the model from the previous turn
            {
                let player = &self.players[active];
                self.model.update_weights(self.param.gamma * q_predict - player.last_q, player.e_trace);
            }

            // update the eligibility trace
            {
                let player = &mut self.players[active];
                player.e_trace *= self.param.lambda * self.param.gamma;
                player.e_trace += &grad;
                player.last_q = q_predict;
            }

            // play the chosen action
            let evs = self.engine.play_action(chosen_action)?;

            for i in 0..2 {
                for ev in &evs[i] {
                    self.players[i].state.on_event(ev);
                }
            }
        }

        // Once the game is over, perform the final update based on the game result.
        let winner = self.engine.winner().expect("must be a winner at game-over phase.");
        let loser = 1 - winner;
        self.model.update_weights(1.0 - self.players[winner].last_q, self.players[winner].e_trace);
        self.model.update_weights(0.0 - self.players[loser].last_q, self.players[loser].e_trace);
        Ok(())
    }
}
