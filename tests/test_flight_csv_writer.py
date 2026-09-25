import csv
import io
from contextlib import closing

import pytest

from haltere.liftoff.visual_brain import QueuedCsvWriter


def test_rows_are_written_in_order_and_drained_on_close():
    f = io.StringIO()
    with closing(QueuedCsvWriter(f)) as writer:
        writer.writerow(['wall', 'x'])
        for k in range(2000):
            writer.writerow([k, k*.5])
    rows = list(csv.reader(io.StringIO(f.getvalue())))
    assert rows[0] == ['wall', 'x'] and len(rows) == 2001 and rows[-1] == ['1999', '999.5']


class Broken(io.StringIO):
    def write(self, text):
        raise OSError('disk full')


def test_a_write_error_is_raised_at_close():
    writer = QueuedCsvWriter(Broken())
    writer.writerow([1, 2])
    with pytest.raises(OSError, match='disk full'):
        writer.close()
