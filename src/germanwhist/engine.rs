pub use cards::card::{Card, Suit, Value};
pub use cards::deck::{Deck};

#[derive(Clone)]
pub struct Action {
    player: usize,
    card: Card
}

#[derive(Clone, Copy)]
pub enum ActionError {
    WrongPlayer(usize),
    MissingCard
}

pub trait PhaseState {
    /// Starting the state
    fn enter(&mut self, player: usize) -> Self;

    /// Indication of whose turn it is
    fn whose_turn(&self) -> usize;

    /// card played already, if any
    fn card_on_board(&self) -> Option<Card>;

    /// perform action on a submitted, return rounds left in this state
    fn on_action(&mut self, engine: &mut Round, action: Action) -> Result<usize, ActionError>;

    /// ending the state
     fn exit(&mut self, engine: &mut Round);
}

pub struct FirstPhase {
    /// card currently played
    played: Option<Card>,

    /// player whose turn it is
    active: usize
}

impl PhaseState for FirstPhase {
    fn enter(&mut self, player: usize) -> Self {
        FirstPhase{ active: player, played: None }
    }

    fn whose_turn(&self) -> usize {
        self.active
    }

    fn card_on_board(&self) -> Option<Card> {
        self.played
    }

    fn on_action(&mut self, engine: &mut Round, action: Action) -> Result<usize, ActionError> {
        if action.player != self.active {
            return Err(ActionError::WrongPlayer(self.active));
        }

        // make sure the player owns the card
        if !engine.hands[action.player].contains(&action.card) {
            return Err(ActionError::MissingCard);
        }

        match self.played {
            /// this is the first card played
            None => {
                self.played = Some(action.card);
                self.active = 1 - self.active;
            },
            /// second card played; this will finish a trick
            Some(card) => {
                let player = engine.player_view_mut(action.player);
            }

        }

    }

    fn exit(&mut self, engine: &mut Round) {
    }

}

pub struct Round {
    /// current deck
    deck: Deck,

    /// hands for each player
    hands: [Vec<Card>; 2],

    /// currently revealed card, if any
    revealed: Option<Card>,

    /// trump suit
    trump: Suit
}

pub struct PlayerViewMut<'a> {
    hand: &'a mut Vec<Card>
}

impl<'a> PlayerViewMut<'a> {
    pub fn has_card(&self, c: Card) -> bool {
        self.hand.contains(&c)
    }

    pub fn has_suit(&self, s: Suit) -> bool {
        self.hand.iter().any(|&card| card.suit == s)
    }
}


impl Round {
    /// Create a new round
    fn new(&mut self) -> Round {
        let mut deck = Deck::new_shuffled();

        let hands = [deck.draw_n(13).ok().unwrap(),
                     deck.draw_n(13).ok().unwrap()];
        let c = deck.draw().ok().unwrap();
        let trump = c.suit;
        let revealed = Some(c);

        Round { deck, hands, revealed, trump }
    }

    pub fn player_view_mut<'a>(&'a mut self, player: usize) -> PlayerViewMut<'a> {
        PlayerViewMut { hand: &mut self.hands[player] }
    }
}
