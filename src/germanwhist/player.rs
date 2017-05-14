///
use hand_belief::HandBelief;
use ndarray::prelude::*;
use std::collections::HashSet;
use germanwhist::engine::GameEvent;
use cards::prelude::*;

/// Representation of current state for learning value function.
pub struct PlayerState {
    // Round-static, implied state
    player_id: usize,
    trump: Suit,

    // Explicit state components
    hand: HashSet<BasicCard>,
    oppo: HandBelief,
    active: usize,
    revealed: Option<BasicCard>,
    leading_card: Option<BasicCard>,
    played_cards: HashSet<BasicCard>,
    score: [usize; 2],

    // state vector
    state_vector: Array<f32, Ix1>,
    suit_order: [Suit; 4],

    // whether the state needs to be recomputed
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
                      played_cards: HashSet::new(),
                      score: [0, 0],
                      suit_order: [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades],
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
                self.played_cards = HashSet::new();
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
                for card in &trick.cards_played {
                    self.played_cards.insert(*card);
                }
            }
        };

        self.dirty = true;
    }

    /// update the state vector
    fn update_state_vector(&mut self) {
        if !self.dirty {
            return;
        }

        self.update_suit_order();
        let mut state_view = self.state_vector.view_mut();
        state_view = {
            let (hand_view, state_view) = state_view.split_at(Axis(0), NUM_BASIC_CARDS);
            Self::cards_to_vector(hand_view, &self.hand, &self.suit_order);

            let (oppo_view, state_view) = state_view.split_at(Axis(0), NUM_BASIC_CARDS);
            Self::oppo_to_vector(oppo_view, &self.oppo, &self.suit_order);

            let (played_view, state_view) = state_view.split_at(Axis(0), NUM_BASIC_CARDS);
            Self::cards_to_vector(played_view, &self.played_cards, &self.suit_order);

            let (revealed_view, state_view) = state_view.split_at(Axis(0), NUM_BASIC_CARDS);
            Self::card_to_vector(revealed_view, &self.revealed, &self.suit_order);

            let (leading_view, state_view) = state_view.split_at(Axis(0), NUM_BASIC_CARDS);
            Self::card_to_vector(leading_view, &self.leading_card, &self.suit_order);

            state_view
        };

        // whose turn it is
        state_view[0] = if self.active == self.player_id { 1.0 } else { -1.0 };

        fn score_to_state(x: usize) -> f32 { x as f32 * 2.0 / 13.0 - 1.0 }

        // current score (scaled to -1.0 -> 1.0)
        state_view[1] = score_to_state(self.score[0]);
        state_view[2] = score_to_state(self.score[1]);
    }

    pub fn state_vector(&mut self) -> ArrayView<f32, Ix1> {
        self.update_state_vector();
        self.state_vector.view()
    }

    /// Return the index of the card in a vector representation
    fn card_index(card: &BasicCard, suit_order: &[Suit]) -> usize {
        card.rank as usize + 13 * suit_order.iter().position(|c| *c == card.suit).unwrap()
    }

    /// Translate a card into a state sub-vector, based on the current suit order
    fn card_to_vector(mut x: ArrayViewMut<f32, Ix1>, card: &Option<BasicCard>, suit_order: &[Suit]) {
        assert_eq!(x.dim(), 52);
        x.fill(0.0);
        if let &Some(ref c) = card {
            x[Self::card_index(c, suit_order)] = 1.0;
        }
    }

    /// Translate the opponent belief set to a state vector
    fn oppo_to_vector(mut x: ArrayViewMut<f32, Ix1>,
                      hb: &HandBelief, suit_order: &[Suit]) {
        assert_eq!(x.dim(), 52);
        for c in BasicCard::all() {
            x[Self::card_index(&c, suit_order)] = hb.p(&c);
        }
    }

    /// Cards to vector
    fn cards_to_vector(mut x: ArrayViewMut<f32, Ix1>,
                       card: &HashSet<BasicCard>, suit_order: &[Suit]) {
        assert_eq!(x.dim(), 52);
        x.fill(0.0);
        for c in card {
            x[Self::card_index(c, suit_order)] = 1.0;
        }
    }

    /// Return the order of the suits as they should be represented in the state vector.
    fn update_suit_order(&mut self)  {
        let mut suits = self.suit_order;
        suits.sort_by_key(|s| {
            // trump comes first
            (if *s == self.trump { 0 } else { 1 },
             // then, highest card count
             self.hand.iter().filter(|c| c.suit == *s).count(),
             // normally ordinal as tie-breaker
             s.ord()) });
        self.suit_order = suits;
    }
}
