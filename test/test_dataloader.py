import pytest
from src.dataset.dataloader import TrajectoryDataset
import torch
from torch.utils.data import DataLoader

CSV_CONTENT = (
    "Frame,Visibility,X,Y\n"
    "0,1,0.1,1.0\n"
    "1,1,0.2,0.9\n"
    "2,0,0.3,0.8\n"
    "3,1,0.4,0.7\n"
    "4,1,0.5,0.6\n"
    "5,0,0.6,0.5\n"
    "6,1,0.7,0.4\n"
    "7,1,0.8,0.3\n"
    "8,1,0.9,0.2\n"
    "9,1,1.0,0.1\n"
)

def test_dataset_loads_csv_correctly(tmp_path):
    csv = tmp_path / "shuttle.csv"
    csv.write_text(CSV_CONTENT)

    ds = TrajectoryDataset(csv_path=str(csv), input_len=1, pred_len=1)

    assert torch.allclose(ds.coords[0], torch.tensor([0.1, 1.0]))
    assert torch.allclose(ds.coords[1], torch.tensor([0.2, 0.9]))
    assert torch.allclose(ds.coords[2], torch.tensor([0.3, 0.8]))
    
    
@pytest.mark.parametrize(
    "input_len,pred_len,expected_len",
    [
        (3, 2, 6),
        (5, 2, 4),
        (1, 1, 9),
    ],
)
def test_dataset_length(tmp_path, input_len, pred_len, expected_len):
    csv = tmp_path / "shuttle.csv"
    csv.write_text(CSV_CONTENT)

    ds = TrajectoryDataset(csv_path=str(csv), input_len=input_len, pred_len=pred_len)
    assert len(ds) == expected_len


@pytest.mark.parametrize(
    "input_len,pred_len,x_shape,y_shape,expected_x,expected_y",
    [
        (
            2,
            1,
            (2, 2),
            (1, 2),
            [[0.2, 0.9], [0.3, 0.8]],
            [[0.4, 0.7]]
        ), # expected input: coords[1:3]
        (
            4,
            2,
            (4, 2),
            (2, 2),
            [[0.2, 0.9], [0.3, 0.8], [0.4, 0.7], [0.5, 0.6]],
            [[0.6, 0.5], [0.7, 0.4]]
        ), # expected input: coords[1:5]
    ],
)
def test_dataset_item_content(tmp_path, input_len, pred_len, x_shape, y_shape, expected_x, expected_y):
    csv = tmp_path / "shuttle.csv"
    csv.write_text(CSV_CONTENT)
    
    ds = TrajectoryDataset(str(csv), input_len=input_len, pred_len=pred_len)

    x, y = ds[1]

    assert x.shape == x_shape
    assert y.shape == y_shape

    expected_x = torch.tensor(expected_x)
    expected_y = torch.tensor(expected_y)

    assert torch.allclose(x, expected_x)
    assert torch.allclose(y, expected_y)


def test_dataloader_batching(tmp_path):
    csv = tmp_path / "ball.csv"
    csv.write_text(CSV_CONTENT)

    ds = TrajectoryDataset(str(csv), input_len=2, pred_len=1)
    loader = DataLoader(ds, batch_size=2)

    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (2, 2, 2)
    assert batch_y.shape == (2, 1, 2)
    

def test_dataset_too_short(tmp_path):
    csv = tmp_path / "ball.csv"
    csv.write_text(
        "Frame,Visibility,X,Y\n"
        "0,1,0.1,0.1\n"
        "1,1,0.2,0.2\n"
    )

    with pytest.raises(ValueError):
        TrajectoryDataset(str(csv), input_len=3, pred_len=2)
