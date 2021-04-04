#![feature(type_ascription)]
#![feature(format_args_capture)]

use std::fmt::{self, Display};
use std::io::{self, Read};
use std::collections::HashSet;
use fairypieces_engine::{
    board::{Transformation, SquareBoardGeometry, Isometry},
    piece::PieceDefinitionIndex,
    victory_conditions::Outcome,
    delta::{GameStateDelta, ReversibleGameStateDelta},
    game::{Game, GameState, PlayerIndex},
    games::international_chess,
    math::{IVec2, IVecComponent},
};
use pgn_reader::{Visitor, Role, San, SanPlus, Square, BufferedReader, CastlingSide, Color};

#[cfg(feature = "concurrency")]
use rayon::prelude::*;

#[cfg(test)]
pub mod tests;

#[derive(Debug)]
pub enum ParseError {
    IoError(io::Error),
    ValidationError(ValidationError),
}

impl From<io::Error> for ParseError {
    fn from(from: io::Error) -> Self {
        ParseError::IoError(from)
    }
}

impl From<ValidationError> for ParseError {
    fn from(from: ValidationError) -> Self {
        ParseError::ValidationError(from)
    }
}

#[derive(Debug, Clone)]
pub enum InvalidMoveType {
    Unimplemented,
    Underspecified,
    Illegal,
}

#[derive(Debug, Clone)]
pub enum ValidationError {
    InvalidMove {
        index: usize,
        mv: String,
        ty: InvalidMoveType,
    },
    UnexpectedOutcome {
        expected: Option<Outcome>,
        evaluated: Option<Outcome>,
    },
}

#[cfg(feature = "concurrency")]
fn validate_pgn_games_parallel(pgn_games: impl Iterator<Item=Result<PgnGame, ParseError>>, initial_state: Option<GameState<SquareBoardGeometry>>) -> impl IndexedParallelIterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    let pgn_games: Vec<_> = pgn_games.collect();

    pgn_games.into_par_iter().map(move |result| {
        let initial_state = &initial_state;

        result.and_then(|pgn_game| {
            pgn_game.validate(initial_state.clone()).map_err(Into::into)
        })
    })
}

fn validate_pgn_games_sequential(pgn_games: impl Iterator<Item=Result<PgnGame, ParseError>>, initial_state: Option<GameState<SquareBoardGeometry>>) -> impl Iterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    pgn_games.map(move |result| {
        let initial_state = &initial_state;

        result.and_then(|pgn_game| {
            pgn_game.validate(initial_state.clone()).map_err(Into::into)
        })
    })
}

fn parse_pgn_games(read: impl Read) -> impl Iterator<Item=Result<PgnGame, ParseError>> {
    let mut reader = BufferedReader::new(read);
    let mut parser = InternationalChessGameVisitor::new();
    let mut critical_error = false;

    std::iter::from_fn(move || {
        if critical_error {
            return None;
        }

        loop {
            match reader.read_game(&mut parser) {
                Ok(None) => {
                    // No more games to parse.
                    return None;
                }
                Ok(Some(None)) => {
                    // Skip over a game with no moves.
                    continue;
                }
                Ok(Some(Some(game))) => {
                    return Some(Ok(game));
                }
                Err(err) => {
                    critical_error = true;
                    return Some(Err(err.into()));
                }
            }
        }
    })
}

/// Creates a sequential iterator of games of International Chess, verifying the moves.
///
/// When `Some(Err(ParseError::IoError))` is returned, the iterator is terminated and no items will
/// follow.
pub fn parse_games_sequential(read: impl Read, initial_state: Option<GameState<SquareBoardGeometry>>) -> impl Iterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    validate_pgn_games_sequential(parse_pgn_games(read), initial_state)
}

/// Creates a parallel iterator of games of International Chess, verifying the moves.
///
/// When `Some(Err(ParseError::IoError))` is returned, the iterator is terminated and no items will
/// follow.
#[cfg(feature = "concurrency")]
pub fn parse_games_parallel(read: impl Read, initial_state: Option<GameState<SquareBoardGeometry>>) -> impl IndexedParallelIterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    validate_pgn_games_parallel(parse_pgn_games(read), initial_state)
}

/// Like `parse_games_sequential`, but collects all games into a `Vec` and fails early,
/// discarding games parsed before the error was encountered.
pub fn parse_games_all_sequential(read: impl Read, initial_state: Option<GameState<SquareBoardGeometry>>) -> Result<Vec<Game<SquareBoardGeometry>>, ParseError> {
    let mut games = Vec::new();

    for game in parse_games_sequential(read, initial_state) {
        games.push(game?);
    }

    Ok(games)
}

/// Like `parse_games_parallel`, but collects all games into a `Vec` and fails early,
/// discarding games parsed before the error was encountered.
#[cfg(feature = "concurrency")]
pub fn parse_games_all_parallel(read: impl Read, initial_state: Option<GameState<SquareBoardGeometry>>) -> Result<Vec<Game<SquareBoardGeometry>>, ParseError> {
    let mut games = Vec::new();

    for game in parse_games_parallel(read, initial_state).collect::<Vec<_>>() {
        games.push(game?);
    }

    Ok(games)
}

/// Returns the first item of the iterator if there are no other follow.
fn just_one<I: Iterator>(mut iter: I) -> Option<I::Item> {
    let first = iter.next()?;

    if iter.next().is_none() {
        Some(first)
    } else {
        None
    }
}

fn just_one_move<I: Iterator>(mut iter: I, index: usize, san_plus: &SanPlus) -> Result<I::Item, ValidationError> {
    let matched_move = iter.next().ok_or_else(|| {
        // No legal moves found.
        ValidationError::InvalidMove {
            index,
            mv: format!("{}", san_plus),
            ty: InvalidMoveType::Illegal,
        }
    })?;

    if let Some(_) = iter.next() {
        // Found another legal move.
        return Err(ValidationError::InvalidMove {
            index,
            mv: format!("{}", san_plus),
            ty: InvalidMoveType::Underspecified,
        });
    }

    Ok(matched_move)
}


fn pgn_outcome_to_fp_outcome(from: pgn_reader::Outcome) -> Outcome {
    match from {
        pgn_reader::Outcome::Decisive { winner: Color::White }
        => Outcome::Decisive { winner: international_chess::PLAYER_WHITE },
        pgn_reader::Outcome::Decisive { winner: Color::Black }
        => Outcome::Decisive { winner: international_chess::PLAYER_BLACK },
        pgn_reader::Outcome::Draw => Outcome::Draw,
    }
}

fn role_to_piece_definition_index(role: Role) -> PieceDefinitionIndex {
    use Role::*;
    use international_chess::*;

    match role {
        Pawn => PIECE_PAWN,
        Knight => PIECE_KNIGHT,
        Bishop => PIECE_BISHOP,
        Rook => PIECE_ROOK,
        Queen => PIECE_QUEEN,
        King => PIECE_KING,
    }
}

fn square_to_tile(square: Square) -> IVec2 {
    let (file, rank) = square.coords();

    [file.into(), rank.into()].into()
}

fn san_to_delta(game: &Game<SquareBoardGeometry>, index: usize, san_plus: SanPlus) -> Result<Option<ReversibleGameStateDelta<SquareBoardGeometry>>, ValidationError> {
    let san = san_plus.san.clone();
    let current_player = game.move_log().current_state().current_player_index();
    let next_player = (current_player + 1) % game.rules().players().get() as PlayerIndex;

    let result = match san {
        San::Normal { role, file, rank, capture, to, promotion } => {
            let src_definition_index = role_to_piece_definition_index(role);
            let dst_definition_index = promotion.map(|role| role_to_piece_definition_index(role))
                .unwrap_or(src_definition_index);
            let src: [Option<IVecComponent>; 2] = [
                file.map(|file| file as IVecComponent),
                rank.map(|rank| rank as IVecComponent),
            ];
            let dst = square_to_tile(to);

            let available_moves = game.available_moves().moves();
            let matched_moves = available_moves.into_iter()
                .filter(|mv| {
                    let delta = mv.delta();
                    // Select changed tiles with matching destination tile
                    let dst_piece = delta.forward().get(dst);

                    if let Some(Some(dst_piece)) = dst_piece {
                        if dst_piece.definition_index() == dst_definition_index
                                && dst_piece.owner() == current_player {
                            dst_piece
                        } else {
                            // Destination piece is not owned by the current player
                            // or it does not match the piece definition (role) played.
                            return false;
                        }
                    } else {
                        // The destination piece is either unaffected or unset.
                        return false;
                    };

                    if !capture && !promotion.is_some() {
                        // A regular move should only affect a piece of a single kind.
                        // This disambiguates castling from rook moves near the king.
                        let affected_piece_kinds = delta.forward().iter()
                            .filter_map(|(_, piece)| piece.as_ref())
                            .map(|piece| piece.definition_index())
                            .collect::<HashSet<_>>()
                            .len();

                        if affected_piece_kinds > 1 {
                            return false;
                        }
                    }

                    let current_srcs = delta.backward().iter()
                        .filter(|(tile, piece)| {
                            piece.as_ref().map(|piece| {
                                piece.definition_index() == src_definition_index
                                    && piece.owner() == current_player
                            }).unwrap_or(false)
                                && src[0].map(|x| x == tile[0]).unwrap_or(true)
                                && src[1].map(|x| x == tile[1]).unwrap_or(true)
                        });

                    just_one(current_srcs).is_some()
                });

            let matched_move = just_one_move(matched_moves, index, &san_plus)?;

            matched_move.delta().clone()
        }
        San::Castle(castling_side) => {
            let current_state = game.move_log().current_state();
            let current_player = current_state.current_player_index();
            let isometry: Isometry<SquareBoardGeometry> =
                if current_player == international_chess::PLAYER_WHITE {
                    Isometry::default()
                } else if current_player == international_chess::PLAYER_BLACK {
                    Isometry::translation([0, 7].into())
                } else {
                    unreachable!();
                };

            let src_king: IVec2 = [4, 0].into();
            let [src_rook, dst_king, dst_rook]: [IVec2; 3] = match castling_side {
                CastlingSide::KingSide => [[7, 0].into(), [6, 0].into(), [5, 0].into()],
                CastlingSide::QueenSide => [[0, 0].into(), [2, 0].into(), [3, 0].into()],
            };
            let mut pattern = GameStateDelta::with_next_player(next_player);
            let move_index = game.move_log().len();

            pattern.set(isometry.apply(src_king), None, move_index);
            pattern.set(isometry.apply(src_rook), None, move_index);
            pattern.set(isometry.apply(dst_king), current_state.tile(game.rules().board(), isometry.apply(src_king))
                .and_then(|piece| piece.get_piece().cloned()), move_index);
            pattern.set(isometry.apply(dst_rook), current_state.tile(game.rules().board(), isometry.apply(src_rook))
                .and_then(|piece| piece.get_piece().cloned()), move_index);

            let available_moves = game.available_moves().moves_from(isometry.apply(src_king));
            let matched_moves = available_moves.into_iter()
                .filter(|available_move| pattern.is_part_of(available_move.delta().forward()))
                .cloned();
            let matched_move = just_one_move(matched_moves, index, &san_plus)?;

            matched_move.delta().clone()
        }
        San::Put { role: _, to: _ } => {
            unimplemented!("\"Put\" moves are not implemented.");
        }
        San::Null => { return Ok(None) }
    };

    Ok(Some(result))
}

#[derive(Clone, Debug, Default)]
pub struct PgnGame {
    moves: Vec<SanPlus>,
    outcome: Option<Outcome>,
}

impl PgnGame {
    pub fn validate(&self, initial_state: Option<GameState<SquareBoardGeometry>>) -> Result<Game<SquareBoardGeometry>, ValidationError> {
        let mut game = if let Some(initial_state) = initial_state {
            Game::new(international_chess::GAME.rules().clone(), initial_state)
        } else {
            international_chess::GAME.clone()
        };

        for (index, san_plus) in self.moves.iter().cloned().enumerate() {
            // println!("Validating move: {:?}", san_plus);
            let current_player = game.move_log().current_state().current_player_index();
            san_to_delta(&game, index, san_plus.clone()).and_then(|delta| {
                if let Some(delta) = delta {
                    game.append_delta(delta).unwrap();
                    Ok(())
                } else {
                    Err(ValidationError::InvalidMove {
                        index,
                        mv: format!("{}", san_plus),
                        ty: InvalidMoveType::Unimplemented,
                    })
                }
            })?;

            if let Some('#') = san_plus.suffix.map(|suffix| suffix.char()) {
                let expected_outcome = Outcome::Decisive { winner: current_player };
                let evaluated_outcome = game.get_outcome().cloned();

                if Some(&expected_outcome) != evaluated_outcome.as_ref() {
                    return Err(ValidationError::UnexpectedOutcome {
                        expected: Some(expected_outcome),
                        evaluated: evaluated_outcome,
                    });
                }
            }
        }

        Ok(game)
    }
}

struct InternationalChessGameVisitor {
    game: Option<PgnGame>,
}

impl InternationalChessGameVisitor {
    fn new() -> Self {
        Self {
            game: None,
        }
    }
}

impl Visitor for InternationalChessGameVisitor {
    type Result = Option<PgnGame>;

    fn end_game(&mut self) -> Self::Result {
        std::mem::replace(self, Self::new()).game
    }

    fn end_headers(&mut self) -> pgn_reader::Skip {
        // Do not skip over moves (which follow after headers).
        pgn_reader::Skip(false)
    }

    fn begin_variation(&mut self) -> pgn_reader::Skip {
        // Skip over variations, which are alternative moves that did not occur.
        pgn_reader::Skip(true)
    }

    fn san(&mut self, san_plus: pgn_reader::SanPlus) {
        if self.game.is_none() {
            self.game = Some(Default::default());
        }

        let game = self.game.as_mut().unwrap();

        game.moves.push(san_plus);
    }

    fn outcome(&mut self, outcome: Option<pgn_reader::Outcome>) {
        if let Some(game) = self.game.as_mut() {
            game.outcome = outcome.map(|outcome| pgn_outcome_to_fp_outcome(outcome));
        }
    }
}
