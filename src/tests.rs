use super::*;
use fairypieces_engine::board::BoardGeometry;
use std::fs::File;
use std::io::Cursor;

#[cfg(feature = "concurrency")]
fn count_games(
    read: impl Read,
    initial_state: Option<GameState<SquareBoardGeometry>>,
) -> (u32, u32) {
    let (total, successful) = parse_games_parallel(read, initial_state)
        .enumerate()
        .map(|(index, game)| {
            if let Err(error) = &game {
                println!("Could not parse game #{index}: {error:?}");
            }

            game.is_ok()
        })
        .fold_with((0, 0), |(mut total, mut successful), result| {
            total += 1;
            if result {
                successful += 1;
            }
            (total, successful)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    (total, successful)
}

#[cfg(not(feature = "concurrency"))]
fn count_games(
    read: impl Read,
    initial_state: Option<GameState<SquareBoardGeometry>>,
) -> (u32, u32) {
    let (total, successful) = parse_games_sequential(read, initial_state)
        .enumerate()
        .map(|(index, game)| {
            if let Err(error) = &game {
                println!("Could not parse game #{index}: {error:?}");
            }

            game.is_ok()
        })
        .fold((0, 0), |(mut total, mut successful), result| {
            total += 1;
            if result {
                successful += 1;
            }
            (total, successful)
        });

    (total, successful)
}

fn test_parse(read: impl Read, initial_state: Option<GameState<SquareBoardGeometry>>) {
    let (total, successful) = count_games(read, initial_state);

    let errors = total - successful;
    let error_percentage = (errors as f64 / total as f64) * 100.0;

    eprintln!("Invalid games: {errors} out of {total} ({error_percentage:.2}%)");

    if errors > 0 {
        panic!(
            "Failed to parse {errors} out of {total} games.",
            errors = errors,
            total = total
        );
    }
}

fn test_parse_one(
    read: impl Read,
    initial_state: Option<GameState<SquareBoardGeometry>>,
) -> Game<SquareBoardGeometry> {
    let game = just_one(parse_games_sequential(read, initial_state)).unwrap();

    game.unwrap_or_else(|err| panic!("{:?}", err))
}

fn test_parse_one_from_str(
    read: &str,
    initial_state: Option<GameState<SquareBoardGeometry>>,
) -> Game<SquareBoardGeometry> {
    test_parse_one(Cursor::new(read), initial_state)
}

fn list_moves(
    game: &mut Game<SquareBoardGeometry>,
    from: Option<IVec2>,
    expected_len: Option<usize>,
) {
    let mut moves: Vec<_> = if let Some(from) = from {
        game.available_moves().moves_from(from).cloned().collect()
    } else {
        game.available_moves().moves().cloned().collect()
    };

    println!("Board:\n{game}", game = SquareBoardGeometry::print(&game));

    if let Some(expected_len) = expected_len {
        assert_eq!(expected_len, moves.len());
    }

    moves.sort_unstable();
    println!("\nMoves:");

    for (index, available_move) in moves.into_iter().enumerate() {
        let game = game
            .clone()
            .append_without_evaluation(available_move.clone())
            .unwrap();

        println!(
            "Possible move #{index}:\n{game}",
            game = SquareBoardGeometry::print(&game)
        );
    }
}

fn test_parse_one_from_str_list(
    read: &str,
    initial_state: Option<GameState<SquareBoardGeometry>>,
    from: Option<IVec2>,
    expected_len: Option<usize>,
) -> Game<SquareBoardGeometry> {
    let mut game = test_parse_one_from_str(read, initial_state);

    list_moves(&mut game, from, expected_len);

    game
}

#[test]
fn validate_resignation() {
    test_parse_one_from_str(
        r#"""
[Termination "Normal"]

1. d4 d5 2. Nf3 Nf6 3. e3 Bf5 4. Nh4 Bg6 5. Nxg6 hxg6 6. Nd2 e6 7. Bd3 Bd6 8. e4 dxe4 9. Nxe4 Rxh2 10. Ke2 Rxh1 11. Qxh1 Nc6 12. Bg5 Ke7 13. Qh7 Nxd4+ 14. Kd2 Qe8 15. Qxg7 Qh8 16. Bxf6+ Kd7 17. Qxh8 Rxh8 18. Bxh8 1-0
    """#,
        None,
    );
}

#[test]
fn validate_checkmate() {
    test_parse_one_from_str(
        r#"""
[Termination "Normal"]

1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0
    """#,
        None,
    );
}

#[test]
fn validate_stalemate() {
    test_parse_one_from_str(
        r#"""
[Termination "Normal"]

1. Nf3 e5 2. Nxe5 f6 3. Nf3 f5 4. d3 b5 5. g3 Bb7 6. Bg2 h6 7. e4 fxe4 8. dxe4 Bxe4 9. Nc3 Bxf3 10. Bxf3 Nf6 11. Bxa8 Qe7+ 12. Be3 Qb4 13. O-O Qxb2 14. Bxa7 Qxc3 15. Qe2+ Kf7 16. Qxb5 c6 17. Qxb8 Qa5 18. Bd4 Qb5 19. Qc7 Ne8 20. Qxd7+ Kg8 21. Qxe8 Qd5 22. c3 Qc4 23. Bxc6 Qf7 24. Qxf7+ Kxf7 25. a4 Kg8 26. a5 Kf7 27. a6 Bc5 28. Bxc5 Ra8 29. Bxa8 Kg6 30. a7 Kh5 31. Bf3+ Kg5 32. a8=Q Kf5 33. Bd4 Kg6 34. c4 Kg5 35. c5 Kg6 36. c6 Kg5 37. c7 Kg6 38. c8=Q Kg5 39. h4+ Kg6 40. h5+ Kg5 41. g4 Kh4 42. Ra5 g6 43. hxg6 h5 44. g7 hxg4 45. Bxg4 1/2-1/2
    """#,
        None,
    );
}

#[test]
fn validate_castle_king_side_valid_no_threat() {
    test_parse_one_from_str(
        r#"""
1. Nf3 Nf6 2. g4 g5 3. Bh3 Bh6 4. O-O O-O
    """#,
        None,
    );
}

#[test]
fn validate_castle_queen_side_valid_no_threat() {
    test_parse_one_from_str(
        r#"""
1. Na3 Na6 2. d4 d5 3. Be3 Be6 4. Qd3 Qd6 5. O-O-O O-O-O
    """#,
        None,
    );
}

#[test]
fn validate_castle_king_side_valid_threat_h_file() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [__, __, __, __, BK, __, __, BR],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, BP, __, WR],
        [__, __, __, __, __, WP, __, BR],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, WK, __, __, WR],
    ]);

    test_parse_one_from_str("1. O-O O-O", Some(state));
}

#[test]
fn validate_castle_queen_side_valid_threat_ab_files() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [BR, __, __, __, BK, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [WR, WR, __, BP, __, __, __, __],
        [BR, BR, __, WP, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [WR, __, __, __, WK, __, __, __],
    ]);

    test_parse_one_from_str("1. O-O-O O-O-O", Some(state));
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 0, mv: \"O-O\", ty: Illegal })")]
fn validate_castle_king_side_invalid_e1_check() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [BK, __, __, __, BR, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, WK, __, __, WR],
    ]);

    test_parse_one_from_str("1. O-O", Some(state));
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 0, mv: \"O-O\", ty: Illegal })")]
fn validate_castle_king_side_invalid_f1_check() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [BK, __, __, __, __, BR, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, WK, __, __, WR],
    ]);

    test_parse_one_from_str("1. O-O", Some(state));
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 0, mv: \"O-O\", ty: Illegal })")]
fn validate_castle_king_side_invalid_g1_check() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [BK, __, __, __, __, __, BR, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, WK, __, __, WR],
    ]);

    test_parse_one_from_str("1. O-O", Some(state));
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 0, mv: \"O-O-O\", ty: Illegal })")]
fn validate_castle_queen_side_invalid_e1_check() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [__, __, __, __, BR, __, __, BK],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [WR, __, __, __, WK, __, __, __],
    ]);

    test_parse_one_from_str("1. O-O-O", Some(state));
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 0, mv: \"O-O-O\", ty: Illegal })")]
fn validate_castle_queen_side_invalid_d1_check() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [__, __, __, BR, __, __, __, BK],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [WR, __, __, __, WK, __, __, __],
    ]);

    test_parse_one_from_str("1. O-O-O", Some(state));
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 0, mv: \"O-O-O\", ty: Illegal })")]
fn validate_castle_queen_side_invalid_c1_check() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [__, __, BR, __, __, __, __, BK],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [WR, __, __, __, WK, __, __, __],
    ]);

    test_parse_one_from_str("1. O-O-O", Some(state));
}

#[test]
fn validate_promotion_to_queen() {
    test_parse_one_from_str(
        r#"""
[Termination "Normal"]

1. h4 g5 2. hxg5 Nf6 3. g6 Bh6 4. g7 Be3 5. gxh8=Q+ Ng8 6. Qxg8# 1-0
    """#,
        None,
    );
}

#[test]
fn validate_promotion_to_rook() {
    test_parse_one_from_str(
        r#"""
[Termination "Normal"]

1. h4 g5 2. hxg5 Nf6 3. g6 Bh6 4. g7 Be3 5. gxh8=R+ Ng8 6. Rxg8# 1-0
    """#,
        None,
    );
}

#[test]
fn validate_en_passant_valid_regular() {
    test_parse_one_from_str(
        r#"""
1. e4 c6 2. e5 f5 3. exf6
    """#,
        None,
    );
}

#[test]
fn validate_en_passant_valid_custom_initial_state() {
    use international_chess::pieces::*;
    let state = international_chess::create_initial_state([
        [BK, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, BP, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, __],
        [__, __, __, __, __, __, __, WP],
        [WK, __, __, __, __, __, __, __],
    ]);

    test_parse_one_from_str_list(
        "1. h4 Kb8 2. h5 g5",
        Some(state),
        Some([7, 4].into()),
        Some(2),
    );
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 6, mv: \"exf6\", ty: Illegal })")]
fn validate_en_passant_invalid_late() {
    test_parse_one_from_str(
        r#"""
1. e4 c6 2. e5 f5 3. Ke2 c5 4. exf6
    """#,
        None,
    );
}

#[test]
#[should_panic(expected = "ValidationError(InvalidMove { index: 4, mv: \"exf6\", ty: Illegal })")]
fn validate_en_passant_invalid_small_steps() {
    test_parse_one_from_str(
        r#"""
1. e4 f6 2. e5 f5 3. exf6
    """#,
        None,
    );
}

#[test]
fn validate_long() {
    test_parse_one_from_str(
        r#"""
1. e4 e5 2. Qf3 Qf6 3. Qxf6 Nxf6 4. Nh3 d6 5. Bb5+ Bd7 6. Bxd7+ Nbxd7 7. f3 Be7 8. d3 O-O 9. Nc3 c6 10. Bg5 h6 11. Bxf6 Bxf6 12. O-O-O a6 13. Na4 Bg5+ 14. f4 exf4 15. Nxg5 hxg5 16. h4 f6 17. hxg5 fxg5 18. Rdg1 b5 19. Nc3 b4 20. Ne2 Ne5 21. Rh5 g4 22. Nd4 Rae8 23. c3 bxc3 24. bxc3 Nxd3+ 25. Kc2 Nf2 26. Nxc6 Rxe4 27. g3 f3 28. Nd4 Nh3 29. Nf5 Re2+ 30. Kb3 Rb8+ 31. Ka3 f2 32. Rh1 Rb7 33. Nxd6 Rbb2 34. Nc4 Rxa2+ 35. Kb3 g6 36. Ra5 Rxa5 37. Nxa5 Kg7 38. Rf1 Ng5 39. Nc4 Nf3 40. Ne3 Nd2+ 41. Ka3 Nxf1 42. Nxg4 Re1 43. Nxf2 Ra1+ 44. Kb2 Ra5 45. c4 Nxg3 46. Nd3 Ne4 47. Kc2 g5 48. Ne1 g4 49. Kd3 Nf6 50. Ng2 g3 51. Kd4 Nh5 52. c5 Rxc5 53. Kxc5 a5 54. Ne3 Kf6 55. Kb5 Ke5 56. Kxa5 Ke4 57. Ng2 Kf3 58. Ne1+ Kf2 59. Nd3+ Ke3 60. Ne1 Kf2 61. Nd3+ Kf1 62. Ne5 g2 63. Nf3 Nf4 64. Nd2+ Ke2 65. Nc4 g1=Q 66. Kb5 Qb1+ 67. Kc5 Kd3 68. Ne5+ Ke4 69. Kd6 Qc1 70. Nc6 Qa3+ 71. Kc7 Qc3 72. Kd6 Ng6 73. Kd7 Ne5+ 74. Kd6 Qxc6+ 75. Ke7 Qd7+ 76. Kf6 Qf5+ 77. Ke7 Ng6+ 78. Ke8 Qf8+ 79. Kd7 Ke5 80. Kc7 Qd6+ 81. Kb7 Ke6 82. Ka7 Kd7 83. Kb7 Kd8 84. Ka7 Qc5+ 85. Kb7 Qc7+ 86. Ka6 Kc8 87. Kb5 Ne5 88. Kb4 Qc4+ 89. Ka5 Qa2+ 90. Kb5 Kc7 91. Kb4 Qc4+ 92. Ka3 Qc1+ 93. Kb4 Kc6 94. Kb3 Kc5 95. Ka4 Qc2+ 96. Ka5 Nc4+ 97. Ka6 Qa4+ 98. Kb7 Qc6+ 99. Kb8 Qe8+ 100. Kb7 Nd6+ 101. Kc7 Qc8# 0-1
    """#,
        None,
    );
}

#[test]
fn validate_promotion_to_rook_list() {
    const PGN_STRING: &str = r#"""
1. h4 g5 2. hxg5 Nf6 3. g6 Bh6 4. g7 Bf4
"""#;
    test_parse_one_from_str_list(PGN_STRING, None, Some([6, 6].into()), Some(8));
}

#[ignore]
#[test]
fn validate_from_file_5000_games() {
    const PATH: &str = "resources/lichess_db_standard_rated_2013-01_truncated_5000.pgn";
    let file = File::open(PATH).unwrap();

    test_parse(file, None);
}

#[ignore]
#[test]
fn validate_from_file_500_games() {
    const PATH: &str = "resources/lichess_db_standard_rated_2013-01_truncated_500.pgn";
    let file = File::open(PATH).unwrap();

    test_parse(file, None);
}

#[ignore]
#[test]
fn validate_from_file_50_games() {
    const PATH: &str = "resources/lichess_db_standard_rated_2013-01_truncated_50.pgn";
    let file = File::open(PATH).unwrap();

    test_parse(file, None);
}
