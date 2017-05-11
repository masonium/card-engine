/// modules for storing player and opponent model
use std::collections::HashMap;
use cards::BasicCard;

///
pub enum CardState {
    /// The card has been played, or the player otherwise definitely
    /// doesn't have the card.
    Played,

    /// The player has the card with the given likelihood.
    Prob(f32),

    /// The player definitely has this card.
    Owns
}

pub struct PlayerBelief {
    probs: HashMap<BasicCard, CardState>,

    relative_priors: HashMap<BasicCard, f32>,

    num_cards: usize,
    dirty: bool
}

impl PlayerBelief {
    pub fn new() -> PlayerBelief {
        let mut probs = HashMap::new();
        let mut priors = HashMap::new();
        for card in BasicCard::all() {
            probs.insert(card.clone(), CardState::Prob(0.25));
            priors.insert(card, 1.0);
        }

        PlayerBelief { probs: probs,
                       relative_priors: priors,
                       num_cards: 0, dirty: true }
    }

    /// Return the probability that the player has the card.
    pub fn has_card(&mut self, card: &BasicCard) -> f32 {
        if self.dirty {
            self.renormalize();
        }

        match self.probs.get(card).expect("All basic cards should be in probability map.") {
            &CardState::Played => 0.0,
            &CardState::Owns => 1.0,
            &CardState::Prob(ref p) => *p
        }
    }

    /// Re-normalize probabilities, preserving beliefs by suit.
    fn renormalize(&mut self) {

    }

    /// Change the prior, based on other information.
    pub fn set_prior(&mut self, card: &BasicCard, prior: f32) {
        self.relative_priors.insert(*card, prior);
        self.dirty = true;
    }

    pub fn random_card_drawn(&mut self) {
        self.num_cards += 1;
        self.dirty = true;
    }

    /// Show that a particular card has been picked up.
    pub fn card_drawn(&mut self, card: &BasicCard) {
        self.num_cards += 1;
        self.probs.insert(*card, CardState::Owns);
    }

    /// Mark that the card has been seen.
    pub fn card_played(&mut self, card: &BasicCard) {
        self.num_cards -= 1;
        self.relative_priors.insert(*card, 0.0);
        self.probs.insert(*card, CardState::Played);
        self.dirty = true;
    }

    /// Mark that card was played by the player.
    pub fn card_seen(&mut self, card: &BasicCard) {
        self.relative_priors.insert(*card, 0.0);
        self.probs.insert(*card, CardState::Played);
        self.dirty = true;
    }
}
