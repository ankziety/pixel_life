#!/usr/bin/env python3

import unittest
import numpy as np
from env import PixelLifeEnv


class TestPixelLifeEnv(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test."""
        self.env = PixelLifeEnv(grid_size=10, max_steps=100, render_mode=None)
    
    def tearDown(self):
        """Clean up after each test."""
        self.env.close()
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        self.assertEqual(self.env.grid_size, 10)
        self.assertEqual(self.env.max_steps, 100)
        self.assertEqual(self.env.action_space.n, 5)
        self.assertEqual(self.env.observation_space.shape, (10, 10, 3))
    
    def test_reset_functionality(self):
        """Test that reset works correctly."""
        obs, info = self.env.reset()
        
        # Check observation shape
        self.assertEqual(obs.shape, (10, 10, 3))
        
        # Check that agent is at center
        expected_pos = np.array([5, 5])
        np.testing.assert_array_equal(self.env.agent_pos, expected_pos)
        
        # Check step count is reset
        self.assertEqual(self.env.step_count, 0)
        
        # Check that grid has some initial cells
        self.assertGreaterEqual(np.sum(self.env.grid), 0)
    
    def test_action_execution(self):
        """Test that actions are executed correctly."""
        self.env.reset()
        initial_pos = self.env.agent_pos.copy()
        
        # Test movement actions
        test_cases = [
            (0, np.array([-1, 0])),  # Move up
            (1, np.array([1, 0])),   # Move down
            (2, np.array([0, -1])),  # Move left
            (3, np.array([0, 1]))    # Move right
        ]
        
        for action, expected_delta in test_cases:
            self.env.reset()
            initial_pos = self.env.agent_pos.copy()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            expected_pos = np.clip(initial_pos + expected_delta, 0, self.env.grid_size - 1)
            np.testing.assert_array_equal(self.env.agent_pos, expected_pos)
    
    def test_cell_placement(self):
        """Test cell placement/removal action."""
        self.env.reset()
        
        # Clear the cell at agent position
        x, y = self.env.agent_pos
        self.env.grid[x, y] = 0
        
        # Place a cell
        obs, reward, terminated, truncated, info = self.env.step(4)
        self.assertEqual(self.env.grid[x, y], 1)
        self.assertGreater(reward, 0)  # Should get positive reward for placing
        
        # Remove the cell
        obs, reward, terminated, truncated, info = self.env.step(4)
        self.assertEqual(self.env.grid[x, y], 0)
        self.assertLess(reward, 0)  # Should get negative reward for removing
    
    def test_cellular_automata_rules(self):
        """Test that cellular automata rules are applied correctly."""
        self.env.reset()
        
        # Create a simple pattern (3 cells in a line)
        self.env.grid.fill(0)
        self.env.grid[5, 4:7] = 1
        self.env.cell_age.fill(0)
        
        # Update cells
        self.env._update_cells()
        
        # Check that middle cell survived and new cells were born
        self.assertEqual(self.env.grid[5, 5], 1)  # Middle cell should survive
        self.assertEqual(self.env.grid[4, 5], 1)  # Cell above should be born
        self.assertEqual(self.env.grid[6, 5], 1)  # Cell below should be born
    
    def test_neighbor_counting(self):
        """Test neighbor counting function."""
        self.env.reset()
        self.env.grid.fill(0)
        
        # Set up a pattern
        self.env.grid[5, 5] = 1
        self.env.grid[4, 4:7] = 1  # 3 neighbors above
        self.env.grid[6, 4:7] = 1  # 3 neighbors below
        
        # Test neighbor count for center cell
        neighbors = self.env._count_neighbors(5, 5)
        self.assertEqual(neighbors, 6)
        
        # Test edge case
        edge_neighbors = self.env._count_neighbors(0, 0)
        self.assertGreaterEqual(edge_neighbors, 0)
        self.assertLessEqual(edge_neighbors, 3)  # Max 3 neighbors at corner
    
    def test_observation_format(self):
        """Test that observations have correct format."""
        obs, info = self.env.reset()
        
        # Check shape
        self.assertEqual(obs.shape, (10, 10, 3))
        
        # Check data types
        self.assertEqual(obs.dtype, np.float32)
        
        # Check value ranges
        self.assertTrue(np.all(obs >= 0))
        self.assertTrue(np.all(obs <= 1))
        
        # Check that agent position is marked in channel 1
        agent_x, agent_y = self.env.agent_pos
        self.assertEqual(obs[agent_x, agent_y, 1], 1.0)
    
    def test_termination_conditions(self):
        """Test environment termination."""
        self.env.reset()
        
        # Run until max steps
        for _ in range(self.env.max_steps):
            obs, reward, terminated, truncated, info = self.env.step(0)
            if terminated:
                break
        
        self.assertTrue(terminated)
        self.assertEqual(self.env.step_count, self.env.max_steps)
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        self.env.reset()
        
        # Test movement reward
        obs, reward, terminated, truncated, info = self.env.step(0)  # Move up
        self.assertIsInstance(reward, (int, float))
        
        # Test that population affects reward
        initial_population = np.sum(self.env.grid)
        self.assertIn('population', info)
        self.assertEqual(info['population'], initial_population)
    
    def test_stability_reward(self):
        """Test stability reward calculation."""
        self.env.reset()
        
        # Set a specific population
        self.env.grid.fill(0)
        self.env.grid[:10, :10] = 1  # 100 cells (in ideal range)
        self.env.step_count = 15  # Past initial steps
        
        stability_reward = self.env._calculate_stability_reward()
        self.assertEqual(stability_reward, 0.5)  # Should get stability bonus
        
        # Test overpopulation penalty
        self.env.grid.fill(1)  # All cells alive (100 > 300 for 10x10 grid)
        stability_reward = self.env._calculate_stability_reward()
        self.assertEqual(stability_reward, -1.0)
    
    def test_boundary_conditions(self):
        """Test agent movement at boundaries."""
        self.env.reset()
        
        # Move agent to corner
        self.env.agent_pos = np.array([0, 0])
        
        # Try to move beyond boundary
        obs, reward, terminated, truncated, info = self.env.step(0)  # Move up
        self.assertEqual(self.env.agent_pos[0], 0)  # Should stay at boundary
        
        obs, reward, terminated, truncated, info = self.env.step(2)  # Move left
        self.assertEqual(self.env.agent_pos[1], 0)  # Should stay at boundary
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = self.env.reset()
            steps = 0
            
            while steps < 50:  # Run for 50 steps
                action = np.random.randint(0, 5)
                obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Environment should be in valid state after each episode
            self.assertEqual(obs.shape, (10, 10, 3))
            self.assertTrue(np.all(obs >= 0))
            self.assertTrue(np.all(obs <= 1))


class TestEnvironmentEdgeCases(unittest.TestCase):
    
    def test_small_grid(self):
        """Test with very small grid size."""
        env = PixelLifeEnv(grid_size=3, max_steps=10)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (3, 3, 3))
        env.close()
    
    def test_large_grid(self):
        """Test with larger grid size."""
        env = PixelLifeEnv(grid_size=50, max_steps=10)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (50, 50, 3))
        env.close()
    
    def test_zero_max_steps(self):
        """Test with zero max steps."""
        env = PixelLifeEnv(grid_size=10, max_steps=1)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        env.close()


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPixelLifeEnv)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEnvironmentEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All {result.testsRun} tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")