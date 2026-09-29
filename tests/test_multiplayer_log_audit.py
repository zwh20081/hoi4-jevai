import unittest
from dev_tools.hoi4.check_multiplayer_logs import compare


def sample(posture=2):
    return ('[[ Launching MULTIPLAYER-game ]]\n'
            f'[12:00:00][1936.1.4.1][effect]: JEV|ACT|GER|posture={posture}\n'
            '[12:30:00][1936.4.4.1][game]: tick\n')


class LogAuditTests(unittest.TestCase):
    def test_identical_empty_logs_are_not_evidence(self):
        self.assertFalse(compare('', '')['log_checks_passed'])

    def test_dates_and_postures_must_match(self):
        self.assertTrue(compare(sample(), sample())['log_checks_passed'])
        self.assertFalse(compare(sample(),sample(3))['log_checks_passed'])
        self.assertFalse(compare(sample(),sample().replace('1936.1.4.1','1936.1.5.1'))['log_checks_passed'])

    def test_new_session_does_not_reuse_old_success(self):
        log=sample()+'[[ Launching MULTIPLAYER-game ]]\n'
        self.assertFalse(compare(log,log)['log_checks_passed'])

    def test_desync_is_reported(self):
        log=sample()+'out of synch\n'
        self.assertFalse(compare(log,log)['log_checks_passed'])


if __name__=='__main__':unittest.main()
