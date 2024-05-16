"""Testing the console module."""

from click import testing
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from src.hough_transform_project import console as console
from src.hough_transform_project.ImageProcessor import ImageProcessor as ImageProcessor


@pytest.fixture
def runner() -> testing.CliRunner:
    """Fixture for creating a Click CLI runner."""
    return testing.CliRunner()


@pytest.mark.e2e
def test_main_succeeds_in_production_env(runner):
    """e2e testing."""
    result = runner.invoke(console.main)
    assert result.exit_code == 0


def test_main_succeeds(runner: testing.CliRunner) -> None:
    """Test that the main function exits with a status code of zero."""
    result: testing.Result = runner.invoke(console.main)
    assert result.exit_code == 0


def test_image_processor_init() -> None:
    """Test ImageProcessor initialization."""
    img_name = 'aaaa'
    img_processor = ImageProcessor(img_name)
    assert img_processor.img_name == img_name
    assert img_processor.max_img_size == 200
    assert img_processor.red_color == (255, 0, 0)


def test_image_processor_resize_to_const() -> None:
    """Test ImageProcessor resize to const."""
    img = Image.open('../images/horizontal_lines.png')
    img_processor = ImageProcessor('')
    new_img = img_processor.resize_to_const(img)
    m = max(new_img.size)
    assert m <= img_processor.max_img_size

    new_img = img_processor.resize_to_const(new_img)
    m = max(new_img.size)
    assert m <= img_processor.max_img_size


def test_image_processor_calc_sums() -> None:
    """Test ImageProcessor calc_sums."""
    img = np.array(Image.open('../images/horizontal_lines.png').convert("L"))
    img_processor = ImageProcessor('')
    res = img_processor.calc_sums(img, 0, 5, img.shape[0])
    assert isinstance(res, np.ndarray)


def test_image_processor_process_image() -> None:
    """Test ImageProcessor process_image."""
    img_processor = ImageProcessor('../images/horizontal_lines.png')
    color_img = img_processor.process_image()
    assert isinstance(color_img, Image.Image)
