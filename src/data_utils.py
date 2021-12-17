# import chord_recognition
import numpy as np
import miditoolkit
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

STRING_MAPPING ={'1':36, '2':35, '3':34, '4':33, '5':32, '6':31}




class CustomNote(object):
    def __init__(self, name, note_on, note_off,
                 velocity, string, fret_value, pitch):
        
        self.name = name
        self.note_on = note_on
        self.note_off = note_off
        self.velocity = velocity
        self.string = string
        self.fret_value = fret_value
        self.pitch = pitch
    def __repr__(self):
        return 'CustomNote(name={}, note_on={}, note_off={}, velocity={}, string={}, fret_value={}, pitch={})'.format(
            self.name, self.note_on, self.note_off, self.velocity, self.string, self.fret_value, self.pitch)

def mapping_string_note_to_pitch_note(correct_string_notes, pitch_note_timing):
    for x in correct_string_notes:
        if 0>= x.start - pitch_note_timing >=-2:
#             print("i found it!",x.start)
            return x
    return None


# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    correct_midi_data = miditoolkit.midi.parser.MidiFile(file_path)


    """ count corret pitch note"""

    note_count = 0

    notes_list= []

    sustain_pop_list = []
    palm_mute_list = []
    hammer_on_list = []
    slide_list = []
    correct_string_notes = []


    scratch_list = []
    slap_list = []
    press_list = []
    stroke_muting_list = []
    downstroke1_list = []
    upstroke1_list = []
    downstroke2_list = []
    upstroke2_list = []
    hit_top_open_list = []
    hit_top_mute_list = []
    hit_rim_list = []
    

    # notes = midi_obj.instruments[0].notes
    if len(correct_midi_data.instruments) == 0:
        return None

    instrument = correct_midi_data.instruments[0]
    note_pool = []
    in_same_time_pool = []
    for note in instrument.notes:
        # pitch notes
        if note.pitch >=40 and note.pitch<=84:
            note_count +=1
            notes_list.append(note)

        # sustain pop 
        if note.pitch == 24:
            sustain_pop_list.append(note)

        # palm mute
        if note.pitch == 26:
            palm_mute_list.append(note)

        # legato slide
        if note.pitch == 28:
            slide_list.append(note)

        # Hammern-on and pull-off
        if note.pitch == 29:
            hammer_on_list.append(note)

        # string notation
        if note.pitch >=31 and note.pitch<=36:
            note_count +=1
            correct_string_notes.append(note)

        # FX sound group
        if note.pitch == 89:
            scratch_list.append(note)
        if note.pitch == 90:
            slap_list.append(note)
        if note.pitch == 91:
            press_list.append(note)
        if note.pitch == 92:
            stroke_muting_list.append(note)
        if note.pitch == 93:
            downstroke1_list.append(note)
        if note.pitch == 94:
            upstroke1_list.append(note)
        if note.pitch == 95:
            downstroke2_list.append(note)
        if note.pitch == 96:
            upstroke2_list.append(note)

        if note.pitch == 101:
            hit_top_open_list.append(note)
        if note.pitch == 102:
            hit_top_mute_list.append(note)
        if note.pitch == 103:
            hit_rim_list.append(note)

           


    # print("pitch note_count",note_count)
    correct_timing_pitch_notes = notes_list

    correct_timing_pitch_notes.sort(key=lambda ts: ts.start)
    correct_string_notes.sort(key=lambda ts: ts.start)
    sustain_pop_list.sort(key=lambda ts: ts.start)
    palm_mute_list.sort(key=lambda ts: ts.start)
    slide_list.sort(key=lambda ts: ts.start)
    hammer_on_list.sort(key=lambda ts: ts.start)

    scratch_list.sort(key=lambda ts: ts.start)
    slap_list.sort(key=lambda ts: ts.start)
    press_list.sort(key=lambda ts: ts.start)
    stroke_muting_list.sort(key=lambda ts: ts.start)
    downstroke1_list.sort(key=lambda ts: ts.start)
    upstroke1_list.sort(key=lambda ts: ts.start)
    downstroke2_list.sort(key=lambda ts: ts.start)
    upstroke2_list.sort(key=lambda ts: ts.start)
    hit_top_open_list.sort(key=lambda ts: ts.start)
    hit_top_mute_list.sort(key=lambda ts: ts.start)
    hit_rim_list.sort(key=lambda ts: ts.start)
    

    

    # print("correct_timing_pitch_notes",len(correct_timing_pitch_notes))
    # print("correct_string_notes",len(correct_string_notes))

    """ deal with pitch note """ 
    custom_note_list = []
    for i, note in enumerate(correct_timing_pitch_notes):
        if i == len(correct_string_notes):
            break
        string_pitch = mapping_string_note_to_pitch_note(correct_string_notes, note.start)


        if string_pitch == None:
            fret_board_place = -1
            string = 0
        else:
            if string_pitch.pitch ==31:   
                fret_board_place = note.pitch - 40 
                string = 6
            elif string_pitch.pitch ==32: 
                fret_board_place = note.pitch - 45
                string = 5
            elif string_pitch.pitch ==33:
                fret_board_place = note.pitch - 50
                string = 4
            elif string_pitch.pitch ==34:
                fret_board_place = note.pitch - 55
                string = 3
            elif string_pitch.pitch ==35:
                fret_board_place = note.pitch - 59
                string = 2
            elif string_pitch.pitch ==36:
                fret_board_place = note.pitch - 64
                string = 1
            if fret_board_place <-1:
                fret_board_place = -1
                
            # else:
            #     fret_board_place = -1
            #     string = 0


        note_start_tick = note.start
        note_end_tick  = note.end

        r = note_start_tick % (correct_midi_data.ticks_per_beat*4)
        # print(r)
        if r >1800:
            # print(i, note.start,note.end)
            interval = (correct_midi_data.ticks_per_beat*4) - r + 1
            note_start_tick = note_start_tick + interval
            note_end_tick = note_end_tick + interval
            if note_start_tick % (correct_midi_data.ticks_per_beat*4) !=1:
                print("fuck",note.start % (correct_midi_data.ticks_per_beat*4), r, interval)


        # note_start_sec = tick_to_time_mapping[note.start]
        # note_end_sec = tick_to_time_mapping[note.end]

        custom_note = CustomNote('pitch_note',note_start_tick, note_end_tick,
                        note.velocity, string, fret_board_place, note.pitch)
        custom_note_list.append(custom_note)
    
    """ deal with tecnique note """ 
    # sustain 
    for i, note in enumerate(sustain_pop_list):
        note_start_tick = note.start
        note_end_tick = note.end
        custom_note = CustomNote('sustain_pop', note_start_tick, note_end_tick,
                        note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
     # palm_mute 
    for i, note in enumerate(palm_mute_list):
        note_start_tick = note.start
        note_end_tick = note.end
        custom_note = CustomNote('palm_mute', note_start_tick, note_end_tick,
                        note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    # slide 
    for i, note in enumerate(slide_list):
        note_start_tick = note.start
        note_end_tick = note.end
        custom_note = CustomNote('slide', note_start_tick, note_end_tick,
                        note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
     # hammer on 
    for i, note in enumerate(hammer_on_list):
        note_start_tick = note.start
        note_end_tick = note.end
        custom_note = CustomNote('hammer_on', note_start_tick, note_end_tick,
                        note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)



    """ deal with FX group note """ 
    for i, note in enumerate(scratch_list):
        custom_note = CustomNote('scratch', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(slap_list):
        custom_note = CustomNote('slap', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(press_list):
        custom_note = CustomNote('press', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(stroke_muting_list):
        custom_note = CustomNote('stroke', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(downstroke1_list):
        custom_note = CustomNote('downstroke1', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(upstroke1_list):
        custom_note = CustomNote('upstroke1', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(downstroke2_list):
        custom_note = CustomNote('downstroke2', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(upstroke2_list):
        custom_note = CustomNote('upstroke2', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(hit_top_open_list):
        custom_note = CustomNote('hit_top_open', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(hit_top_mute_list):
        custom_note = CustomNote('hit_top_mute', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    for i, note in enumerate(hit_rim_list):
        custom_note = CustomNote('hit_rim', note.start, note.end, note.velocity, None, None, note.pitch)
        custom_note_list.append(custom_note)
    



    custom_note_list.sort(key=lambda x: x.note_on)

    def quantize_items(items, ticks=120):
        # grid
        grids = np.arange(0, items[-1].note_on, ticks, dtype=int)
        # process
        for item in items:
            index = np.argmin(abs(grids - item.note_on))
            shift = grids[index] - item.note_on
            item.note_on += shift
            item.note_off += shift
        return items 
    custom_note_list = quantize_items(custom_note_list)
    custom_note_list.sort(key=lambda x: (x.note_on, -x.pitch))
    
    # for i in custom_note_list:
    #     print(i)
    # exit()



    return custom_note_list

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.note_on)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.note_on >= db1) and (item.note_on < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)





# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    pos_flag = 1
    temp_pos = None
    for i in range(len(groups)):
        if 'pitch_note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            
            if 'grooving' in item.name :
                events.append(Event(
                    name=item.name, 
                    time=None,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                continue

            
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.note_on))
            
           

            events.append(Event(
                name='Position', 
                time=item.note_on,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.note_on)))
      

            if item.name == 'pitch_note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.note_on, 
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.note_on,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.note_off - item.note_on
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.note_on,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
                # string
                events.append(Event(
                    name='String',
                    time=item.note_on,
                    value=item.string,
                    text='{}'.format(item.string)))
                # fret_value
                events.append(Event(
                    name='Fret value',
                    time=item.note_on,
                    value=item.fret_value,
                    text='{}'.format(item.string)))
            elif item.name == 'slap':
                events.append(Event(name='Slap', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'press':
                events.append(Event(name='Press', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'stroke':
                events.append(Event(name='Stroke', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'downstroke1':
                events.append(Event(name='Downstroke1', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'upstroke1':
                events.append(Event(name='Upstroke1', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'downstroke2':
                events.append(Event(name='Downstroke2', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'upstroke2':
                events.append(Event(name='Upstroke2', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'hit_top_open':
                events.append(Event(name='Hit Top Open', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))
            elif item.name == 'hit_top_mute':
                events.append(Event(name='Hit Top Mute', time=item.note_on, value=item.pitch, text='{}'.format(item.pitch)))


    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        count = 0
        # count _
        get_word = word2event.get(word)
        for char in get_word:
            if char =='_':
                count += 1
        if count ==2:
            replace_idx = get_word.index('_', 9)
            get_word = get_word[:replace_idx] + '/' + get_word[replace_idx+1:]

        
        event_name, event_value = get_word.split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

def write_midi(words, word2event, output_path, prompt_path=None, stringInfo=True):
    events = word_to_event(words, word2event)
    print("fucks!!!!!!!!!!!!!!!!!")
    # print("events",events)
    # get downbeat and note (no time)
    flag = 0
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events)-3):
        if events[i].name == 'Bar':
            flag = 1
        if flag:
            

            try:
    
                if events[i].name == 'Position' and \
                    events[i+1].name == 'Note Velocity' and \
                    events[i+2].name == 'Note On' and \
                    events[i+3].name == 'Note Duration':
                    
                    # start time and end time from position
                    position = int(events[i].value.split('/')[0]) - 1
                    # velocity
                    index = int(events[i+1].value)
                    velocity = int(DEFAULT_VELOCITY_BINS[index])
                    # pitch
                    pitch = int(events[i+2].value)
                    # duration
                    index = int(events[i+3].value)
                    duration = DEFAULT_DURATION_BINS[index]
                    # adding
                    temp_notes.append([position, velocity, pitch, duration])
                    if  events[i+4].name == 'String' and events[i+5].name == 'Fret value': 
                        string_pitch = int(events[i+4].value)
                        if string_pitch !=0:   
                            temp_notes.append([position, 30, STRING_MAPPING[str(string_pitch)], 60])
                
                elif events[i].name == 'Position' and events[i+1].name == 'Scratch':
                    position = int(events[i].value.split('/')[0]) - 1
                    # pitch
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Slap':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Press':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Stroke':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Downstroke1':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Upstroke1':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Downstroke2':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Upstroke2':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Hit Top Open':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Hit Top Mute':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Hit Rim':
                    position = int(events[i].value.split('/')[0]) - 1
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                
                # elif events[i].name == 'Position' and events[i+1].name == 'Sustain pop':
                #     position = int(events[i].value.split('/')[0]) - 1
                #     # pitch
                #     pitch = int(events[i+1].value)
                #     temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Palm Mute':
                    position = int(events[i].value.split('/')[0]) - 1
                    # pitch 
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Slide':
                    position = int(events[i].value.split('/')[0]) - 1
                    # pitch
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
                elif events[i].name == 'Position' and events[i+1].name == 'Hammer On':
                    position = int(events[i].value.split('/')[0]) - 1
                    # pitch
                    pitch = int(events[i+1].value)
                    temp_notes.append([position, 30, pitch, 60])
            except:
                pass
            
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    last_position = -1
    current_bar = 0
    for note in temp_notes:
        position, velocity, pitch, duration = note
        if position < last_position:
            current_bar += 1
        # position (start time)
        current_bar_st = current_bar * ticks_per_bar
        current_bar_et = (current_bar + 1) * ticks_per_bar
        flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
        st = flags[position]
        # duration (end time)
        et = st + duration
        notes.append(miditoolkit.Note(velocity, pitch, st, et))
        # record bar
        last_position = position
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        last_position = -1
        current_bar = 0
        for chord in temp_chords:
            position, value = chord
            if position <= last_position:
                current_bar += 1
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            chords.append([st, value])
            # record bar
            last_position = position
    # get specific time for tempos
    tempos = []
    last_position = -1
    current_bar = 0
    for tempo in temp_tempos:
        position, value = tempo
        if position < last_position:
            current_bar += 1
        # position (start time)
        current_bar_st = current_bar * ticks_per_bar
        current_bar_et = (current_bar + 1) * ticks_per_bar
        flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
        st = flags[position]
        tempos.append([int(st), value])
        # record bar
        last_position = position
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)