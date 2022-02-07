use crate::germanwhist::{self, Action, ActionError, Round, PlayerState, ScoringRules};

use crate::learning::model::{LearningModel, LearningModelError};
use ndarray::prelude::*;
use rand::{Rng, thread_rng};

pub struct SarsaLambdaParameters {
    lambda: f32,
    gamma: f32,
    eps: f32,
}

impl Default for SarsaLambdaParameters {
    fn default() -> Self {
        SarsaLambdaParameters {
            gamma: 1.0,
            lambda: 0.8,
            eps: 0.01,
        }
    }
}
pub struct SarsaPlayer {
    pub state: PlayerState,
    e_trace: Array<f32, Ix1>,
    last_q: f32,
}

impl SarsaPlayer {
    pub fn new(state: PlayerState, model_size: usize) -> SarsaPlayer {
        let e_trace = Array::zeros(model_size);

        SarsaPlayer {
            state,
            e_trace,
            last_q: 0.0,
        }
    }

    /// Choose an epsilon-greedy action.
    fn epsilon_greedy_action<M: LearningModel>(&self,
                                               model: &M,
                                               eps: f32,
                                               actions: &[Action],
                                               mut sa: ArrayViewMut<f32, Ix1>)
                                               -> Action {
        assert_eq!(sa.dim(), PlayerState::state_action_size());

        // epsilon-greedy state-choosing
        let mut rng = thread_rng();
        let r = rng.next_f32();

        // choose a random action with probability epsilon
        if r < eps {
            let random_action = rng.choose(actions)
                .expect("must have positive number of actions");
            self.state
                .state_action_vector(sa.view_mut(), true, Some(random_action));
            *random_action
        } else {
            self.state.state_action_vector(sa.view_mut(), true, None);
            self.greedy_action(model, actions, sa)
        }
    }

    /// Fill in the state action input `sa` most beneficial action of those provided in the
    /// current state, returning the Q-value at that state.
    pub fn greedy_action<M: LearningModel>(&self,
                                           model: &M,
                                           actions: &[Action],
                                           mut sa: ArrayViewMut<f32, Ix1>)
                                           -> Action {
        let (max_action, _) = actions
            .iter()
            .zip(actions
                     .iter()
                     .map(|a| {
                              self.state
                                  .state_action_vector(sa.view_mut(), false, Some(a));
                              model.evaluate_q(&sa.view())
                          }))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("action list should not be empty");

        self.state.state_action_vector(sa, false, Some(max_action));
        *max_action
    }
}

pub struct SarsaLambda<M: LearningModel> {
    players: [SarsaPlayer; 2],
    model: M,
    engine: germanwhist::Round,
    param: SarsaLambdaParameters,
}

impl<M: LearningModel> SarsaLambda<M> {
    pub fn new(rules: ScoringRules,
               model: M,
               param: SarsaLambdaParameters)
               -> Result<SarsaLambda<M>, LearningModelError> {
        let players = [SarsaPlayer::new(PlayerState::new(0), model.num_parameters()),
                       SarsaPlayer::new(PlayerState::new(1), model.num_parameters())];

        if PlayerState::state_size() + PlayerState::action_size() != model.input_size() {
            return Err(LearningModelError::MismatchedSize);
        }

        Ok(SarsaLambda {
               players,
               engine: Round::new(rules),
               model,
               param,
           })
    }

    pub fn current_model(&self) -> &M {
        &self.model
    }

    pub fn train_on_episode(&mut self, dual_train: bool) -> Result<(), ActionError> {
        // start a new round
        let ev = self.engine.start_round(None);

        for (i, evs) in ev.iter().enumerate() {
            self.players[i].e_trace.fill(0.0);
            for e in evs {
                self.players[i].state.on_event(e)
            }
        }

        let mut player_action = Array::zeros(PlayerState::state_action_size());
        let mut grad = Array::zeros(self.model.num_parameters());

        // Evaluate the episode, tracking and updating the trace for each player
        while !self.engine.is_game_over() {
            // Get the next action based on whose turn it is.
            let active = self.engine.active_player();

            // choose the epsilon-greedy action for that player.
            let possible_actions = self.engine.possible_actions();
            let chosen_action = self.players[active].epsilon_greedy_action(&self.model,
                                                                           self.param.eps,
                                                                           &possible_actions,
                                                                           player_action.view_mut());

            // evaluate the gradient for the state-action pair
            let q_predict = self.model
                .evaluate_q_grad(&player_action.view(), grad.view_mut());

            // update the model from the previous turn
            if dual_train || active == 0 {
                {
                    let player = &self.players[active];
                    self.model
                        .update_weights(self.param.gamma * q_predict - player.last_q,
                                        &player.e_trace);
                }

                // update the eligibility trace
                {
                    let player = &mut self.players[active];
                    player.e_trace *= self.param.lambda * self.param.gamma;
                    player.e_trace += &grad;
                    player.last_q = q_predict;
                }
            }
            // play the chosen action
            let _ = self.engine.play_action(chosen_action)?;

            for (i, evs) in ev.iter().enumerate() {
                for ev in evs {
                    self.players[i].state.on_event(ev);
                }
            }
        }

        // Once the game is over, perform the final update based on the game result.
        let winner = self.engine.winner().expect("must be a winner at game-over phase.");
        let loser = 1 - winner;
        self.model.update_weights(1.0 - self.players[winner].last_q, &self.players[winner].e_trace);
        self.model.update_weights(0.0 - self.players[loser].last_q, &self.players[loser].e_trace);
        Ok(())
    }
}
