/// modules for storing player and opponent model
use std::collections::HashMap;
use cards::{BasicCard, Rank, Suit};

///
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CardState {
    /// The card has been played, or the player otherwise definitely
    /// doesn't have the card.
    Played,

    /// The player has the card with the given likelihood.
    Prob(f32),

    /// The player definitely has this card.
    Owns
}

impl CardState {
    // Return the probability that the player owns a card in this
    // state.
    pub fn p(&self) -> f32 {
        match self {
            &CardState::Played => 0.0,
            &CardState::Owns => 1.0,
            &CardState::Prob(ref p) => *p
        }
    }
}

pub struct PlayerBelief {
    probs: HashMap<BasicCard, CardState>,

    relative_priors: HashMap<BasicCard, f32>,

    num_cards: usize,
}

fn print_card_map(map: &HashMap<BasicCard, CardState>) {
    use self::CardState::*;
    for rank in Rank::iterator() {
        print!("| ");
        for suit in Suit::iterator() {
            let bc = BasicCard { rank: *rank, suit: *suit };
            print!("{}{}: ", rank, suit);
            match *map.get(&bc).unwrap() {
                Played => print!("----"),
                Owns => print!("MINE"),
                Prob(p) => print!("{:.2}", p)
            }
            print!(" | ");
        }
        println!("");
    }

}

impl PlayerBelief {
    pub fn new() -> PlayerBelief {
        let mut probs = HashMap::new();
        let mut priors = HashMap::new();
        for card in BasicCard::all() {
            probs.insert(card.clone(), CardState::Prob(0.0));
            priors.insert(card, 0.0);
        }

        PlayerBelief { probs: probs,
                       relative_priors: priors,
                       num_cards: 0 }
    }

    /// print probabilities
    pub fn dump_probabilities(&self) {
        print_card_map(&self.probs);
    }

    /// Return the probability that the player has the card.
    pub fn has_card(&mut self, card: &BasicCard) -> f32 {
        self.probs.get(card).expect("All basic cards should be in probability map.").p()
    }

    /// Re-normalize probabilities w/rt priors
    fn renormalize(&mut self) {
        use self::CardState::*;

        // sum up total prior for unknown cars
        let ps: f32 = self.probs.iter().map(|(k, v)| match v {
            &Prob(_) => *self.relative_priors.get(&k).unwrap(),
            _ => 0.0,
        }).sum();

        let num_owned: f32 = self.probs.values()
            .filter_map(|v| if *v == CardState::Owns { Some(1.0) } else { None }).sum();

        let remaining = self.num_cards as f32 - num_owned;

        for card in BasicCard::all() {
            if let &mut Prob(mut p) = self.probs.get_mut(&card).unwrap() {
                // re-compute the posterior and up-date the prior
                p = {
                    let prior = self.relative_priors.get(&card).unwrap();
                    prior * remaining / ps
                };
                self.relative_priors.insert(card, if ps > 0.0 { p } else { 1.0 });
            }
        }

    }

    /// Change the prior, based on other information.
    // pub fn set_prior(&mut self, card: &BasicCard, prior: f32) {
    //     self.relative_priors.insert(*card, prior);
    //     self.renormalize();
    // }

    pub fn random_card_drawn(&mut self) {
        self.num_cards += 1;
        self.renormalize();
    }

    /// establish that the suit is empty
    pub fn empty_suit(&mut self, suit: Suit) {
        for r in Rank::iterator() {
            let card = BasicCard { rank: *r, suit };
            self.relative_priors.insert(card, 0.0);
        }
        self.renormalize();
    }

    /// Show that a particular card has been picked up.
    pub fn card_drawn(&mut self, card: &BasicCard) {
        self.num_cards += 1;
        self.probs.insert(*card, CardState::Owns);
        self.renormalize();
    }

    /// Mark that the card has been seen.
    pub fn card_played(&mut self, card: &BasicCard) {
        self.num_cards -= 1;
        self.relative_priors.insert(*card, 0.0);
        self.probs.insert(*card, CardState::Played);
        self.renormalize();
    }

    /// Mark that card was played by the player.
    pub fn card_seen(&mut self, card: &BasicCard) {
        self.relative_priors.insert(*card, 0.0);
        self.probs.insert(*card, CardState::Played);
        self.renormalize();
    }
}
