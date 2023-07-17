from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner import db
from pathlib import Path
import shutil

if __name__ == "__main__":
    aligner = PretrainedAligner(
        corpus_directory=Path('/Users/markharvilla/Downloads/librispeech-test-samples/mfa-test'),
        dictionary_path=Path('/Users/markharvilla/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict'),
        acoustic_model_path=Path('/Users/markharvilla/Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip'),
        speaker_characters=0,
        fine_tune=False
    )
    print('Instantiated PretrainedAligner')

    aligner.align()

    print('Copying new sound file')
    shutil.copy(
        '/Users/markharvilla/Downloads/librispeech-test-samples/input-16k.wav',
        '/Users/markharvilla/Downloads/librispeech-test-samples/mfa-test'
    )

    print('Inserting data into database')
    with aligner.session() as session:
        new_file = db.File(
            id=7,
            name='input-16k',
            relative_path='',
            modified=False
        )
        session.add(new_file)

        new_speaker = db.Speaker(
            id=2,
            name='mark',
            dictionary_id=1
        )
        session.add(new_speaker)

        new_sound_file = db.SoundFile(
            file_id=7,
            sound_file_path='/Users/markharvilla/Downloads/librispeech-test-samples/mfa-test/input-16k.wav',
            format='.wav',
            sample_rate=16000,
            duration=1.379,
            num_channels=1,
            sox_string=''
        )
        session.add(new_sound_file)

        new_utterance = db.Utterance(
            id=7,
            begin=0,
            end=1.379,
            channel=0,
            text='YEAH BABY I MISS YOU',
            normalized_text='yeah baby i miss you',
            oovs='',
            in_subset=False,
            ignored=False,
            file_id=7,
            speaker_id=2
        )
        session.add(new_utterance)
        session.commit()

        result = session.query(db.Utterance).filter(db.Utterance.id == 7).first()

        print('Running align_one_utterance')
        aligner.align_one_utterance(utterance=result, session=session)
