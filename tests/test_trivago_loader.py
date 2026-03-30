"""Tests for the Trivago hotel search DDC dataset loader."""

import pytest
import torch
import numpy as np

from econirl.datasets.trivago_search import (
    load_trivago_sessions,
    build_trivago_mdp,
    build_trivago_panel,
    build_trivago_features,
    build_trivago_transitions,
    _map_action_type,
    ACTION_BROWSE,
    ACTION_REFINE,
    ACTION_CLICKOUT,
    ACTION_ABANDON,
    N_STATES,
    N_ACTIONS,
    ABSORBING_STATE,
)


DATA_PATH = "/Volumes/Expansion/datasets/trivago-2019/train.csv"
N_TEST_SESSIONS = 100


@pytest.fixture(scope="module")
def sessions_df():
    return load_trivago_sessions(data_path=DATA_PATH, n_sessions=N_TEST_SESSIONS)


@pytest.fixture(scope="module")
def mdp_dict(sessions_df):
    return build_trivago_mdp(sessions_df)


@pytest.fixture(scope="module")
def panel(mdp_dict):
    return build_trivago_panel(mdp_dict)


# ---------- Action type mapping ----------


class TestActionTypeMapping:
    def test_browse_actions(self):
        assert _map_action_type("interaction item image") == ACTION_BROWSE
        assert _map_action_type("interaction item info") == ACTION_BROWSE
        assert _map_action_type("interaction item rating") == ACTION_BROWSE
        assert _map_action_type("interaction item deals") == ACTION_BROWSE

    def test_refine_actions(self):
        assert _map_action_type("filter selection") == ACTION_REFINE
        assert _map_action_type("change of sort order") == ACTION_REFINE
        assert _map_action_type("search for destination") == ACTION_REFINE
        assert _map_action_type("search for poi") == ACTION_REFINE
        assert _map_action_type("search for item") == ACTION_REFINE

    def test_clickout_action(self):
        assert _map_action_type("clickout item") == ACTION_CLICKOUT


# ---------- Loading sessions ----------


class TestLoadSessions:
    def test_returns_dataframe(self, sessions_df):
        import polars as pl
        assert isinstance(sessions_df, pl.DataFrame)

    def test_correct_columns(self, sessions_df):
        required = {"user_id", "session_id", "timestamp", "step",
                     "action_type", "device"}
        assert required.issubset(set(sessions_df.columns))

    def test_n_sessions_limit(self, sessions_df):
        n_unique = sessions_df["session_id"].n_unique()
        assert n_unique == N_TEST_SESSIONS


# ---------- MDP construction ----------


class TestBuildMDP:
    def test_required_keys(self, mdp_dict):
        required_keys = {
            "all_states", "all_actions", "all_next_states",
            "session_ids", "n_states", "n_actions",
            "state_names", "action_names",
        }
        assert required_keys == set(mdp_dict.keys())

    def test_n_states(self, mdp_dict):
        assert mdp_dict["n_states"] == N_STATES  # 37

    def test_n_actions(self, mdp_dict):
        assert mdp_dict["n_actions"] == N_ACTIONS  # 4

    def test_states_in_range(self, mdp_dict):
        states = mdp_dict["all_states"]
        assert all(0 <= s <= ABSORBING_STATE for s in states)

    def test_actions_in_range(self, mdp_dict):
        actions = mdp_dict["all_actions"]
        assert all(0 <= a <= 3 for a in actions)

    def test_next_states_in_range(self, mdp_dict):
        next_states = mdp_dict["all_next_states"]
        assert all(0 <= ns <= ABSORBING_STATE for ns in next_states)

    def test_abandon_present(self, mdp_dict):
        """Sessions ending without clickout should have an abandon action."""
        actions = mdp_dict["all_actions"]
        assert ACTION_ABANDON in actions

    def test_clickout_transitions_to_absorbing(self, mdp_dict):
        """Every clickout action should transition to the absorbing state."""
        for s, a, ns in zip(
            mdp_dict["all_states"],
            mdp_dict["all_actions"],
            mdp_dict["all_next_states"],
        ):
            if a == ACTION_CLICKOUT:
                assert ns == ABSORBING_STATE, (
                    f"Clickout at state {s} should go to absorbing ({ABSORBING_STATE}), "
                    f"got {ns}"
                )

    def test_abandon_transitions_to_absorbing(self, mdp_dict):
        """Every abandon action should transition to the absorbing state."""
        for s, a, ns in zip(
            mdp_dict["all_states"],
            mdp_dict["all_actions"],
            mdp_dict["all_next_states"],
        ):
            if a == ACTION_ABANDON:
                assert ns == ABSORBING_STATE

    def test_lengths_consistent(self, mdp_dict):
        n = len(mdp_dict["all_states"])
        assert len(mdp_dict["all_actions"]) == n
        assert len(mdp_dict["all_next_states"]) == n
        assert len(mdp_dict["session_ids"]) == n

    def test_state_names_length(self, mdp_dict):
        assert len(mdp_dict["state_names"]) == mdp_dict["n_states"]


# ---------- Panel construction ----------


class TestBuildPanel:
    def test_num_individuals(self, panel):
        # Should have exactly N_TEST_SESSIONS trajectories
        assert panel.num_individuals == N_TEST_SESSIONS

    def test_trajectory_lengths(self, panel):
        """Each trajectory should have at least 1 step."""
        for traj in panel:
            assert len(traj) >= 1

    def test_trajectory_types(self, panel):
        """Trajectory tensors should be long dtype."""
        for traj in panel:
            assert traj.states.dtype == torch.long
            assert traj.actions.dtype == torch.long
            assert traj.next_states.dtype == torch.long

    def test_states_in_range_panel(self, panel):
        all_states = panel.get_all_states()
        assert (all_states >= 0).all()
        assert (all_states <= ABSORBING_STATE).all()

    def test_terminal_trajectories(self, panel):
        """Last action in each trajectory should be clickout or abandon."""
        for traj in panel:
            last_action = traj.actions[-1].item()
            assert last_action in (ACTION_CLICKOUT, ACTION_ABANDON), (
                f"Last action should be terminal, got {last_action}"
            )


# ---------- Feature matrix ----------


class TestBuildFeatures:
    def test_shape(self):
        features = build_trivago_features()
        assert features.shape == (N_STATES, N_ACTIONS, 4)

    def test_browse_has_negative_step_cost(self):
        features = build_trivago_features()
        # For mid/late step buckets, browse step_cost should be negative
        # State with step_bucket=1 (mid): s = 1 * 12 + 0 * 3 + 0 = 12
        assert features[12, ACTION_BROWSE, 0] < 0

    def test_browse_indicator(self):
        features = build_trivago_features()
        # Browse action should have browse_indicator = -1
        assert features[0, ACTION_BROWSE, 1] == -1.0
        # Non-browse actions should have browse_indicator = 0
        assert features[0, ACTION_REFINE, 1] == 0.0
        assert features[0, ACTION_CLICKOUT, 1] == 0.0

    def test_clickout_has_positive_indicator(self):
        features = build_trivago_features()
        # Clickout should have clickout_indicator = 1 for all non-absorbing states
        for s in range(N_STATES - 1):
            assert features[s, ACTION_CLICKOUT, 3] == 1.0

    def test_abandon_is_zero(self):
        features = build_trivago_features()
        # Abandon (baseline) should be all zeros
        for s in range(N_STATES - 1):
            assert (features[s, ACTION_ABANDON] == 0).all()

    def test_absorbing_state_all_zeros(self):
        features = build_trivago_features()
        assert (features[ABSORBING_STATE] == 0).all()


# ---------- Transition matrix ----------


class TestBuildTransitions:
    def test_shape(self, mdp_dict):
        transitions = build_trivago_transitions(mdp_dict)
        assert transitions.shape == (N_ACTIONS, N_STATES, N_STATES)

    def test_rows_sum_to_one(self, mdp_dict):
        transitions = build_trivago_transitions(mdp_dict)
        row_sums = transitions.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_non_negative(self, mdp_dict):
        transitions = build_trivago_transitions(mdp_dict)
        assert (transitions >= 0).all()

    def test_clickout_to_absorbing(self, mdp_dict):
        """Clickout from any state should transition to absorbing."""
        transitions = build_trivago_transitions(mdp_dict)
        for s in range(N_STATES):
            assert transitions[ACTION_CLICKOUT, s, ABSORBING_STATE] == 1.0

    def test_abandon_to_absorbing(self, mdp_dict):
        """Abandon from any state should transition to absorbing."""
        transitions = build_trivago_transitions(mdp_dict)
        for s in range(N_STATES):
            assert transitions[ACTION_ABANDON, s, ABSORBING_STATE] == 1.0

    def test_absorbing_self_loop(self, mdp_dict):
        """Absorbing state should self-loop for all actions."""
        transitions = build_trivago_transitions(mdp_dict)
        for a in range(N_ACTIONS):
            assert transitions[a, ABSORBING_STATE, ABSORBING_STATE] == 1.0
