use super::engine::{Action, ScoringRules, ActionError};
use super::state::{GameState};

pub trait GamePhase {
    fn possible_actions(&self, gs: &GameState) -> Vec<Action>;

    /// perform action on a submitted, return rounds left in this state
    fn on_action(&mut self, gs: &mut GameState, rules: &ScoringRules, action: Action) -> Result<usize, ActionError>;

    fn format(&self, gs: &GameState) -> String;

    /// ending the state
    fn transition(&mut self, gs: &mut GameState) -> Option<Box<GamePhase>>;
}

pub struct PlayingPhase;

impl GamePhase for PlayingPhase {
    /// Available actions
    fn possible_actions(&self, gs: &GameState) -> Vec<Action> {
        let view = gs.player_view(gs.active);

        view.playable_cards().into_iter().map(|c| Action {player: gs.active, card: c}).collect()
    }

    fn on_action(&mut self, gs: &mut GameState, rules: &ScoringRules, action: Action) -> Result<usize, ActionError> {
        if action.player != gs.active {
            return Err(ActionError::WrongPlayer(gs.active));
        }

        // make sure the player owns the card
        if !gs.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        if gs.played.is_none() {
            /// this is the first card played
            gs.player_view_mut(action.player).remove_card(&action.card)?;

            gs.played = Some(action.card);
            gs.active = 1 - gs.active;
        } else {
            let leading_card = gs.played.take().expect("on_action: already checked !is_none");
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

            let follow = gs.active;
            let lead = 1 - gs.active;

            let winner = if gs.score_hand(&leading_card, &action.card).expect("cards must be distinct") {
                lead
            } else {
                follow
            };
            let loser = 1 - winner;

            // hand-building phase
            if gs.revealed.is_some() {
                /// Give players their new cards
                let r = gs.revealed.take().expect("must be a revealed card");
                gs.player_view_mut(winner).add_card(r);

                let draw = gs.draw().expect("must have a card left after trick");
                gs.player_view_mut(loser).add_card(draw);

                // Draw a new card, if any
                if gs.deck.num_cards_left() > 0 {
                    gs.revealed = gs.draw();
                }

                gs.increment_score(winner, rules.0);
            }
            // scoring phase
            else {
                gs.increment_score(winner, rules.1);
            }

            gs.active = winner;
            gs.rounds_left -= 1;
        }

        Ok(gs.rounds_left)
    }

    fn format(&self, gs: &GameState) -> String {
        match (&gs.revealed, &gs.played) {
            (&None, _) => {
                format!("End of Phase.")
            },
            (&Some(ref reveal), &Some(ref card)) => {
                format!("Playing for {}\nPlayer {} played {}, Player {} to respond.",
                       reveal, 2 - gs.active, card, gs.active + 1)
            },
            (&Some(ref reveal), &None) => {
                format!("Playing for {}\nPlayer {} to open",
                       reveal, gs.active + 1)
            }
        }
    }



    /// Once we're done playing, finish.
    fn transition(&mut self, _: &mut GameState) -> Option<Box<GamePhase>> {
        Some(Box::new(GameOverPhase {}))
    }

}


pub struct GameOverPhase;

impl GamePhase for GameOverPhase {
    fn possible_actions(&self, _: &GameState) -> Vec<Action> {
        Vec::new()
    }

    fn on_action(&mut self, _: &mut GameState, _: &ScoringRules, _: Action) -> Result<usize, ActionError> {
        Err(ActionError::GameOver)
    }

    fn format(&self, _: &GameState) -> String {
        format!("Game Over.")
    }

    fn transition(&mut self, _: &mut GameState) -> Option<Box<GamePhase>> {
        Some(Box::new(GameOverPhase{}))
    }
}
