#![feature(type_ascription)]
#![feature(format_args_capture)]

use std::io::{self, Read};
use std::collections::HashSet;
use fairypieces_engine::{
    board::{Transformation, SquareBoardGeometry, Isometry},
    piece::Piece,
    game::{Game, GameStateDelta, Outcome},
    games::international_chess,
    math::{IVec2, IVecComponent},
};
use pgn_reader::{Visitor, Role, San, SanPlus, Square, BufferedReader, CastlingSide, Color};

#[cfg(feature = "concurrency")]
use rayon::prelude::*;

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
fn validate_pgn_games_parallel(pgn_games: impl Iterator<Item=Result<PgnGame, ParseError>>) -> impl IndexedParallelIterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    let pgn_games: Vec<_> = pgn_games.collect();

    pgn_games.into_par_iter().map(|result| {
        result.and_then(|pgn_game| {
            pgn_game.validate().map_err(Into::into)
        })
    })
}

fn validate_pgn_games_sequential(pgn_games: impl Iterator<Item=Result<PgnGame, ParseError>>) -> impl Iterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    pgn_games.map(|result| {
        result.and_then(|pgn_game| {
            pgn_game.validate().map_err(Into::into)
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
pub fn parse_games_sequential(read: impl Read) -> impl Iterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    validate_pgn_games_sequential(parse_pgn_games(read))
}

/// Creates a parallel iterator of games of International Chess, verifying the moves.
///
/// When `Some(Err(ParseError::IoError))` is returned, the iterator is terminated and no items will
/// follow.
#[cfg(feature = "concurrency")]
pub fn parse_games_parallel(read: impl Read) -> impl IndexedParallelIterator<Item=Result<Game<SquareBoardGeometry>, ParseError>> {
    validate_pgn_games_parallel(parse_pgn_games(read))
}

/// Like `parse_games_sequential`, but collects all games into a `Vec` and fails early,
/// discarding games parsed before the error was encountered.
pub fn parse_games_all_sequential(read: impl Read) -> Result<Vec<Game<SquareBoardGeometry>>, ParseError> {
    let mut games = Vec::new();

    for game in parse_games_sequential(read) {
        games.push(game?);
    }

    Ok(games)
}

/// Like `parse_games_parallel`, but collects all games into a `Vec` and fails early,
/// discarding games parsed before the error was encountered.
#[cfg(feature = "concurrency")]
pub fn parse_games_all_parallel(read: impl Read) -> Result<Vec<Game<SquareBoardGeometry>>, ParseError> {
    let mut games = Vec::new();

    for game in parse_games_parallel(read).collect::<Vec<_>>() {
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


fn pgn_outcome_to_fp_outcome(from: pgn_reader::Outcome) -> Outcome {
    match from {
        pgn_reader::Outcome::Decisive { winner: Color::White }
        => Outcome::Decisive { winner: international_chess::PLAYER_WHITE },
        pgn_reader::Outcome::Decisive { winner: Color::Black }
        => Outcome::Decisive { winner: international_chess::PLAYER_BLACK },
        pgn_reader::Outcome::Draw => Outcome::Draw,
    }
}

fn role_to_piece_definition_index(role: Role) -> usize {
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

fn san_to_delta(game: &Game<SquareBoardGeometry>, index: usize, san_plus: SanPlus) -> Result<Option<GameStateDelta<SquareBoardGeometry>>, ValidationError> {
    let san = san_plus.san.clone();
    let current_player = game.move_log().current_state().currently_playing_player_index;
    let next_player = (current_player + 1) % game.rules().players().get();

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
            let mut matched_moves = available_moves.into_iter()
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

            let matched_move = matched_moves.next().ok_or_else(|| {
                // No legal moves found.
                ValidationError::InvalidMove {
                    index,
                    mv: format!("{}", san_plus),
                    ty: InvalidMoveType::Illegal,
                }
            })?;

            if let Some(_) = matched_moves.next() {
                // Found another legal move.
                return Err(ValidationError::InvalidMove {
                    index,
                    mv: format!("{}", san_plus),
                    ty: InvalidMoveType::Underspecified,
                });
            }

            matched_move.delta().forward().clone()
        }
        San::Castle(castling_side) => {
            let current_state = game.move_log().current_state();
            let current_player = current_state.currently_playing_player_index;
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
            let mut result = GameStateDelta::with_next_player(next_player);

            result.set(isometry.apply(src_king), None);
            result.set(isometry.apply(src_rook), None);
            result.set(isometry.apply(dst_king), current_state.tile(game.rules().board(), isometry.apply(src_king))
                .and_then(|piece| piece.get_piece().map(Piece::clone_moved)));
            result.set(isometry.apply(dst_rook), current_state.tile(game.rules().board(), isometry.apply(src_rook))
                .and_then(|piece| piece.get_piece().map(Piece::clone_moved)));

            result
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
    pub fn validate(&self) -> Result<Game<SquareBoardGeometry>, ValidationError> {
        let mut game = international_chess::GAME.clone();

        for (index, san_plus) in self.moves.iter().cloned().enumerate() {
            // println!("Validating move: {:?}", san_plus);
            let current_player = game.move_log().current_state().currently_playing_player_index;
            san_to_delta(&game, index, san_plus.clone()).and_then(|delta| {
                if let Some(delta) = delta {
                    game.normalize_and_append_delta(delta).unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::fs::File;
    use fairypieces_engine::board::BoardGeometry;

    #[cfg(feature = "concurrency")]
    fn count_games(read: impl Read) -> (u32, u32) {
        let (total, successful) = parse_games_parallel(read).enumerate().map(|(index, game)| {
            if let Err(error) = &game {
                println!("Could not parse game #{index}: {error:?}");
            }

            game.is_ok()
        }).fold_with((0, 0), |(mut total, mut successful), result| {
            total += 1;
            if result { successful += 1; }
            (total, successful)
        }).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

        (total, successful)
    }

    #[cfg(not(feature = "concurrency"))]
    fn count_games(read: impl Read) -> (u32, u32) {
        let (total, successful) = parse_games_sequential(read).enumerate().map(|(index, game)| {
            if let Err(error) = &game {
                println!("Could not parse game #{index}: {error:?}");
            }

            game.is_ok()
        }).fold((0, 0), |(mut total, mut successful), result| {
            total += 1;
            if result { successful += 1; }
            (total, successful)
        });

        (total, successful)
    }

    fn test_parse(read: impl Read) {
        let (total, successful) = count_games(read);

        let errors = total - successful;
        let error_percentage = (errors as f64 / total as f64) * 100.0;

        eprintln!("Invalid games: {errors} out of {total} ({error_percentage:.2}%)");

        if errors > 0 {
            panic!();
        }
    }

    #[test]
    fn validate_resignation() {
        const PGN_STRING: &str = r#"""
[Event "Rated Classical game"]
[Site "https://lichess.org/a9tcp02g"]
[White "Desmond_Wilson"]
[Black "savinka59"]
[Result "1-0"]
[UTCDate "2012.12.31"]
[UTCTime "23:04:12"]
[WhiteElo "1654"]
[BlackElo "1919"]
[WhiteRatingDiff "+19"]
[BlackRatingDiff "-22"]
[ECO "D04"]
[Opening "Queen's Pawn Game: Colle System, Anti-Colle"]
[TimeControl "480+2"]
[Termination "Normal"]

1. d4 d5 2. Nf3 Nf6 3. e3 Bf5 4. Nh4 Bg6 5. Nxg6 hxg6 6. Nd2 e6 7. Bd3 Bd6 8. e4 dxe4 9. Nxe4 Rxh2 10. Ke2 Rxh1 11. Qxh1 Nc6 12. Bg5 Ke7 13. Qh7 Nxd4+ 14. Kd2 Qe8 15. Qxg7 Qh8 16. Bxf6+ Kd7 17. Qxh8 Rxh8 18. Bxh8 1-0
"""#;

        test_parse(Cursor::new(PGN_STRING));
    }

    #[test]
    fn validate_checkmate() {
        const PGN_STRING: &str = r#"""
[Event "Rated Classical game"]
[Site "https://lichess.org/j1dkb5dw"]
[White "BFG9k"]
[Black "mamalak"]
[Result "1-0"]
[UTCDate "2012.12.31"]
[UTCTime "23:01:03"]
[WhiteElo "1639"]
[BlackElo "1403"]
[WhiteRatingDiff "+5"]
[BlackRatingDiff "-8"]
[ECO "C00"]
[Opening "French Defense: Normal Variation"]
[TimeControl "600+8"]
[Termination "Normal"]

1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0
"""#;

        test_parse(Cursor::new(PGN_STRING));
    }

    #[test]
    fn validate_promotion_to_queen() {
        const PGN_STRING: &str = r#"""
[Termination "Normal"]

1. h4 g5 2. hxg5 Nf6 3. g6 Bh6 4. g7 Be3 5. gxh8=Q+ Ng8 6. Qxg8#
"""#;

        test_parse(Cursor::new(PGN_STRING));
    }

    #[test]
    fn validate_promotion_to_rook() {
        const PGN_STRING: &str = r#"""
[Termination "Normal"]

1. h4 g5 2. hxg5 Nf6 3. g6 Bh6 4. g7 Be3 5. gxh8=R+ Ng8 6. Rxg8#
"""#;

        test_parse(Cursor::new(PGN_STRING));
    }

    #[test]
    fn validate_promotion_to_rook_list() {
        const PGN_STRING: &str = r#"""
[Termination "Normal"]

1. h4 g5 2. hxg5 Nf6 3. g6 Bh6 4. g7 Bf4
"""#;
        let move_from: IVec2 = [6, 6].into();

        let games = parse_games_all_sequential(Cursor::new(PGN_STRING)).unwrap();
        let mut game = games[0].clone();

        println!("\nMoves:");

        game.evaluate();

        for (index, available_move) in game.available_moves().moves_from(move_from).enumerate() {
            let mut game = game.clone();

            game.append(available_move.clone()).unwrap();

            println!("Possible move #{index}:\n{game}", game=SquareBoardGeometry::print(&game));
        }
    }

    #[test]
    fn validate_long() {
        const PGN_STRING: &str = r#"""
1. e4 e5 2. Qf3 Qf6 3. Qxf6 Nxf6 4. Nh3 d6 5. Bb5+ Bd7 6. Bxd7+ Nbxd7 7. f3 Be7 8. d3 O-O 9. Nc3 c6 10. Bg5 h6 11. Bxf6 Bxf6 12. O-O-O a6 13. Na4 Bg5+ 14. f4 exf4 15. Nxg5 hxg5 16. h4 f6 17. hxg5 fxg5 18. Rdg1 b5 19. Nc3 b4 20. Ne2 Ne5 21. Rh5 g4 22. Nd4 Rae8 23. c3 bxc3 24. bxc3 Nxd3+ 25. Kc2 Nf2 26. Nxc6 Rxe4 27. g3 f3 28. Nd4 Nh3 29. Nf5 Re2+ 30. Kb3 Rb8+ 31. Ka3 f2 32. Rh1 Rb7 33. Nxd6 Rbb2 34. Nc4 Rxa2+ 35. Kb3 g6 36. Ra5 Rxa5 37. Nxa5 Kg7 38. Rf1 Ng5 39. Nc4 Nf3 40. Ne3 Nd2+ 41. Ka3 Nxf1 42. Nxg4 Re1 43. Nxf2 Ra1+ 44. Kb2 Ra5 45. c4 Nxg3 46. Nd3 Ne4 47. Kc2 g5 48. Ne1 g4 49. Kd3 Nf6 50. Ng2 g3 51. Kd4 Nh5 52. c5 Rxc5 53. Kxc5 a5 54. Ne3 Kf6 55. Kb5 Ke5 56. Kxa5 Ke4 57. Ng2 Kf3 58. Ne1+ Kf2 59. Nd3+ Ke3 60. Ne1 Kf2 61. Nd3+ Kf1 62. Ne5 g2 63. Nf3 Nf4 64. Nd2+ Ke2 65. Nc4 g1=Q 66. Kb5 Qb1+ 67. Kc5 Kd3 68. Ne5+ Ke4 69. Kd6 Qc1 70. Nc6 Qa3+ 71. Kc7 Qc3 72. Kd6 Ng6 73. Kd7 Ne5+ 74. Kd6 Qxc6+ 75. Ke7 Qd7+ 76. Kf6 Qf5+ 77. Ke7 Ng6+ 78. Ke8 Qf8+ 79. Kd7 Ke5 80. Kc7 Qd6+ 81. Kb7 Ke6 82. Ka7 Kd7 83. Kb7 Kd8 84. Ka7 Qc5+ 85. Kb7 Qc7+ 86. Ka6 Kc8 87. Kb5 Ne5 88. Kb4 Qc4+ 89. Ka5 Qa2+ 90. Kb5 Kc7 91. Kb4 Qc4+ 92. Ka3 Qc1+ 93. Kb4 Kc6 94. Kb3 Kc5 95. Ka4 Qc2+ 96. Ka5 Nc4+ 97. Ka6 Qa4+ 98. Kb7 Qc6+ 99. Kb8 Qe8+ 100. Kb7 Nd6+ 101. Kc7 Qc8# 0-1
"""#;

        test_parse(Cursor::new(PGN_STRING));
    }

    #[ignore]
    #[test]
    fn validate_from_file_5000_games() {
        const PATH: &str = "resources/lichess_db_standard_rated_2013-01_truncated_5000.pgn";
        let file = File::open(PATH).unwrap();

        test_parse(file);
    }

    #[ignore]
    #[test]
    fn validate_from_file_500_games() {
        const PATH: &str = "resources/lichess_db_standard_rated_2013-01_truncated_500.pgn";
        let file = File::open(PATH).unwrap();

        test_parse(file);
    }

    #[ignore]
    #[test]
    fn validate_from_file_50_games() {
        const PATH: &str = "resources/lichess_db_standard_rated_2013-01_truncated_50.pgn";
        let file = File::open(PATH).unwrap();

        test_parse(file);
    }
}
