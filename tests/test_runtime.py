import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from runtime.inference import preprocess_image, run_fdi_inference
from runtime.model import postprocess_fdi_prediction


class RuntimeTests(unittest.TestCase):
    def test_unique_fdi_preserves_adjacent_teeth(self):
        prediction = {'scores': torch.tensor([.7, .95, .9]),
                      'labels': torch.tensor([1, 1, 2]),
                      'boxes': torch.tensor([[0., 0., 12., 12.]] * 3),
                      'masks': torch.ones((3, 1, 16, 16))}
        result = postprocess_fdi_prediction(prediction, .5, None, True)
        self.assertEqual(result['labels'].tolist(), [1, 2])
        torch.testing.assert_close(result['scores'], torch.tensor([.95, .9]))

    def test_letterbox_and_restoration_keep_original_coordinates(self):
        image = Image.new('RGB', (40, 20), 'white')
        tensor, original, geometry = preprocess_image(image, (100, 100))
        self.assertEqual(tuple(tensor.shape), (3, 100, 100))
        self.assertEqual(geometry['offset_y'], 25)
        class FakeModel:
            def __call__(self, images):
                mask = torch.zeros((1, 1, 100, 100))
                mask[:, :, 25:75] = 1
                return [{'scores': torch.tensor([.9]), 'labels': torch.tensor([1]),
                         'boxes': torch.tensor([[0., 25., 100., 75.]]), 'masks': mask}]
        teeth = run_fdi_inference(FakeModel(), tensor, geometry, torch.device('cpu'))
        self.assertEqual(teeth[0]['fdi'], 11)
        self.assertEqual((teeth[0]['x1'], teeth[0]['y1'], teeth[0]['x2'], teeth[0]['y2']), (0., 0., 40., 20.))
        self.assertEqual(teeth[0]['mask'].shape, (20, 40))
        self.assertTrue(np.all(teeth[0]['mask']))

    def test_no_detections_is_valid(self):
        prediction = {'scores': torch.tensor([.2]), 'labels': torch.tensor([1]),
                      'boxes': torch.zeros((1, 4)), 'masks': torch.zeros((1, 1, 5, 5))}
        result = postprocess_fdi_prediction(prediction, .5, None, True)
        self.assertEqual(len(result['labels']), 0)

    def test_import_does_not_load_conditions_or_training_pipeline(self):
        self.assertNotIn('pathology_detection', sys.modules)
        self.assertNotIn('instance_segmentation', sys.modules)
        self.assertFalse(any(name.startswith('dentilin_conditions') for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
