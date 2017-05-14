///
use hand_belief::HandBelief;
use ndarray::prelude::*;
use std::collections::HashSet;
use germanwhist::engine::GameEvent;
use cards::prelude::*;

/// Representation of current state for learning value function.
pub struct PlayerState {
    player_id: usize,
    hand: HashSet<BasicCard>,
    oppo: HandBelief,
    state_vector: Array<f32, Ix1>,
    active: usize,
    trump: Suit,
    revealed: Option<BasicCard>,
    leading_card: Option<BasicCard>,
    score: [usize; 2],

    /// whether the state needs to be recomputed
    dirty: bool
}

impl PlayerState {
    pub fn new(id: usize) -> PlayerState {
        PlayerState { player_id: id,
                      hand: HashSet::new(),
                      oppo: HandBelief::new(),
                      state_vector: Array::zeros(52),
                      active: 0,
                      trump: Suit::Hearts,
                      revealed: None,
                      leading_card: None,
                      score: [0, 0],
                      dirty: true}
    }

    fn hand_from_slice(&mut self, v: &[BasicCard]) {
        self.hand.clear();
        for card in v {
            self.hand.insert(*card);
        }
    }

    /// Update the state vector in response to a game action.
    pub fn on_event(&mut self, ev: &GameEvent) {
        use GameEvent::*;
        match ev {
            &Start(ref start) => {
                self.hand_from_slice(&start.hand);
                self.oppo.clear();
                self.oppo.random_cards_drawn(13);
                self.trump = start.trump;
                self.active = start.starting_player;
                self.revealed = Some(start.revealed);
                self.leading_card = None;
                self.score = [0, 0];
            },
            &Action(ref action) => {
                if action.player == self.player_id {
                    self.hand.remove(&action.card);
                } else {
                    self.oppo.card_played(&action.card);

                    // check if we can rule out suits
                    if let Some(p) = self.leading_card {
                        if p.suit != action.card.suit {
                            self.oppo.empty_suit(p.suit);
                        }
                    }
                }

                // mark the leading card, if it exists
                if self.leading_card.is_none() {
                    self.leading_card = Some(action.card)
                }
            },
            &Card(card) => {
                if card.player == self.player_id {
                    self.hand.insert(card.card.expect("Player always knows what card that player gets."));
                } else {
                    match card.card {
                        Some(ref c) => self.oppo.card_drawn(c),
                        None => self.oppo.random_cards_drawn(1)
                    }
                }
            },

            &Trick(ref trick) => {
                self.leading_card = None;
                self.revealed = trick.revealed;
                self.active = trick.active_player;
                self.score = trick.score;
            }
        };

        self.dirty = true;
    }

    /// update the state vector
    fn update_state_vector(&mut self) {
        if !self.dirty {
            return;
        }

        let suits = self.suit_order();

    }

    /// Return the order of the suits as they should be represented in the state vector.
    fn suit_order(&self) -> [Suit; 4] {
        let mut suits = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
        suits.sort_by_key(|s| {
            // trump comes first
            (if *s == self.trump { 0 } else { 1 },
             // then, highest card count
             self.hand.iter().filter(|c| c.suit == *s).count(),
             // normally ordinal as tie-breaker
             s.ord()) });
        suits
    }
}
