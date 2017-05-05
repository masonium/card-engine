use std::fmt::{self, Display, Formatter};
use super::engine::{Action, ActionError};
use super::state::{GameState};
use cards::{BasicCard};


pub trait GamePhase: Display {
    fn possible_actions(&self, gs: &GameState) -> Vec<Action>;

    /// perform action on a submitted, return rounds left in this state
    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError>;

    /// ending the state
    fn transition(&mut self, gs: &mut GameState) -> Option<Box<GamePhase>>;
}

pub struct FirstPhase {
    /// card currently played
    played: Option<BasicCard>,

    /// player whose turn it is
    active: usize,

    rounds_left: usize,

    // currently revealed card, if any
    revealed: Option<BasicCard>,
}

impl FirstPhase {
    pub fn new(first_card: BasicCard, player: usize) -> Self {
        FirstPhase { played: None, active: player,
                     rounds_left: 13,
                     revealed: Some(first_card) }
    }
}

impl Display for FirstPhase {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match (&self.revealed, &self.played) {
            (&None, _) => {
                write!(f, "End of Phase.")
            },
            (&Some(ref reveal), &Some(ref card)) => {
                write!(f, "Playing for {}\nPlayer {} played {}, Player {} to respond.",
                       reveal, 2 - self.active, card, self.active + 1)
            },
            (&Some(ref reveal), &None) => {
                write!(f, "Playing for {}\nPlayer {} to open",
                       reveal, self.active + 1)
            }
        }
    }
}

impl GamePhase for FirstPhase {
    fn possible_actions(&self, gs: &GameState) -> Vec<Action> {
        let view = gs.player_view(self.active);

        match &self.played {
            // Second player must follow suit, if possible.
            &Some(ref c) if view.has_suit(&c.suit) =>
                view.iter().filter(|x| x.suit == c.suit)
                .map(|c| Action { player: self.active, card: c.clone()}).collect()
                ,
            // Otherwise, can play anything
            _ => view.iter().map(|c| Action { player: self.active, card: c.clone()}).collect(),
        }
    }


    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError> {
        if action.player != self.active {
            return Err(ActionError::WrongPlayer(self.active));
        }

        // make sure the player owns the card
        if !gs.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        if self.played.is_none() {
            /// this is the first card played
            gs.player_view_mut(action.player).remove_card(&action.card)?;

            self.played = Some(action.card);
            self.active = 1 - self.active;
        } else {
            let leading_card = self.played.take().unwrap();

            {
                let mut player = gs.player_view_mut(action.player);

                // If the player has the suit, the card must match
                if player.has_suit(&leading_card.suit) {
                    if action.card.suit != leading_card.suit {
                        return Err(ActionError::NotFollowingSuit);
                    }
                }

                player.remove_card(&action.card)?;
            }

            let follow = self.active;
            let lead = 1 - self.active;

            let winner = if gs.score_hand(&leading_card, &action.card).unwrap() {
                lead
            } else {
                follow
            };
            let loser = 1 - winner;

            /// Give players their new cards
            let r = self.revealed.take().expect("must be a revealed card");
            gs.player_view_mut(winner).add_card(r);

            let draw = gs.draw().expect("must have a card left after trick");
            gs.player_view_mut(loser).add_card(draw);

            // Draw a new card, if any
            if gs.deck.num_cards_left() > 0 {
                self.revealed = gs.draw();
            }

            self.active = winner;
            self.rounds_left -= 1;
        }

        Ok(self.rounds_left)
    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<GamePhase>> {
        Some(Box::new(SecondPhase::new(self.active)))
    }

}


impl Display for SecondPhase {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.rounds_left == 0 {
            write!(f, "end of phase")
        } else {
            match self.played {
                Some(ref card) => {
                    write!(f, "layer {} played {}, Player {} to respond.",
                           1 - self.active, card, self.active)
                },
                None => {
                    write!(f, "Player {} to open",
                           1 - self.active)
                }
            }
        }
    }
}

pub struct SecondPhase {
    /// card currently played
    played: Option<BasicCard>,

    /// player whose turn it is
    active: usize,

    /// rounds left
    rounds_left: usize
}

impl SecondPhase {
    pub fn new(player: usize) -> Self {
        SecondPhase{ active: player, played: None, rounds_left: 13 }
    }
}

impl GamePhase for SecondPhase {
    fn possible_actions(&self, gs: &GameState) -> Vec<Action> {
        let view = gs.player_view(self.active);

        match &self.played {
            // Second player must follow suit, if possible.
            &Some(ref c) if view.has_suit(&c.suit) =>
                view.iter().filter(|x| x.suit == c.suit)
                .map(|c| Action { player: self.active, card: c.clone()}).collect()
                ,
            // Otherwise, can play anything
            _ => view.iter().map(|c| Action { player: self.active, card: c.clone()}).collect(),
        }
    }

    fn on_action(&mut self, gs: &mut GameState, action: Action) -> Result<usize, ActionError> {
        if action.player != self.active {
            return Err(ActionError::WrongPlayer(self.active));
        }

        // make sure the player owns the card
        if !gs.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        if self.played.is_none() {
            /// this is the first card played
            gs.player_view_mut(action.player).remove_card(&action.card)?;

            self.played = Some(action.card);
            self.active = 1 - self.active;
            return Ok(self.rounds_left);
        } else {
            /// second card played; this will finish a trick
            let leading_card = self.played.take().unwrap();
            {
                let mut player = gs.player_view_mut(action.player);

                // If the player has the suit, the card must match
                if player.has_suit(&leading_card.suit) {
                    if action.card.suit != leading_card.suit {
                        return Err(ActionError::NotFollowingSuit);
                    }
                }

                player.remove_card(&action.card)?;
            }

            let follow = self.active;
            let lead = 1 - self.active;

            let winner = if gs.score_hand(&leading_card, &action.card).unwrap() {
                lead
            } else {
                follow
            };

            // The winner of the trick score a point and plays the next hand.
            gs.increment_score(winner, 1);

            self.active = winner;
            self.rounds_left -= 1;
        }

        Ok(self.rounds_left)

    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<GamePhase>> {
        Some(Box::new(GameOverPhase{}))
    }

}

pub struct GameOverPhase;

impl Display for GameOverPhase {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Game Over.")
    }
}

impl GamePhase for GameOverPhase {
    fn possible_actions(&self, _: &GameState) -> Vec<Action> {
        Vec::new()
    }

    fn on_action(&mut self, _: &mut GameState, _: Action) -> Result<usize, ActionError> {
        Err(ActionError::GameOver)
    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<GamePhase>> {
        Some(Box::new(GameOverPhase{}))
    }
}
