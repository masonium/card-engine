use cards::{BasicCard};

use super::state::GameState;
use super::phase::{GamePhase, PlayingPhase};

#[derive(Clone, Debug)]
pub struct Action {
    pub player: usize,
    pub card: BasicCard
}

#[derive(Clone, Copy, Debug)]
pub enum ActionError {
    WrongPlayer(usize),
    MissingCard,
    NotFollowingSuit,
    GameOver
}

pub struct Round {
    state: GameState,

    phase: Option<Box<GamePhase>>
}

impl Round {
    pub fn new() -> Round {
        let (state, card) = GameState::new();
        let phase: Option<Box<GamePhase>> = Some(Box::new(PlayingPhase::new(card, 0)));
        Round { state, phase: phase }
    }

    pub fn get_state(&self) -> &GameState {
        &self.state
    }

    pub fn possible_actions(&self) -> Vec<Action> {
        self.phase.as_ref().unwrap().possible_actions(&self.state)
    }

    pub fn get_phase(&self) -> &GamePhase {
        use std::borrow::Borrow;
        self.phase.as_ref().unwrap().borrow()
    }

    pub fn play_action(&mut self, action: Action) -> Result<(), ActionError> {
        let rl = self.phase.as_mut().unwrap().on_action(&mut self.state, action)?;

        if rl == 0 {
            self.phase = self.phase.as_mut().unwrap().transition(&mut self.state);
        }

        Ok(())
    }
}
