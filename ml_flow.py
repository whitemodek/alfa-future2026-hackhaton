"""Stateless row-local network-flow features; no labels or fitted statistics."""
import numpy as np
import pandas as pd


def add_flow_features(frame):
    required = {'spkts', 'dpkts', 'sbytes', 'dbytes', 'proto', 'service', 'state'}
    if not required <= set(frame):
        raise ValueError('Flow features require packet/byte counts and proto/service/state')
    if any(str(c).startswith('__flow_') for c in frame):
        raise ValueError('Input collides with reserved flow feature names')
    result = frame.copy()
    categorical = []

    def number(name):
        return pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan).clip(-1e15, 1e15)

    for left, right, name in (('spkts', 'dpkts', 'packet'), ('sbytes', 'dbytes', 'byte'),
            ('sload', 'dload', 'load'), ('sinpkt', 'dinpkt', 'interval'), ('sjit', 'djit', 'jitter'),
            ('smean', 'dmean', 'mean_packet')):
        if {left, right} <= set(frame):
            a, b = number(left), number(right)
            result['__flow_' + name + '_asymmetry'] = (a - b) / (a.abs() + b.abs() + 1.)
            result['__flow_' + name + '_log_ratio'] = np.log1p(a.clip(lower=0)) - np.log1p(b.clip(lower=0))
    for numerator, denominator, name in (
            ('sloss', 'spkts', 'source_loss_fraction'), ('dloss', 'dpkts', 'destination_loss_fraction'),
            ('ct_srv_src', 'ct_src_ltm', 'source_service_concentration'),
            ('ct_srv_dst', 'ct_dst_ltm', 'destination_service_concentration')):
        if {numerator, denominator} <= set(frame):
            result['__flow_' + name] = number(numerator) / (number(denominator).abs() + 1.)
    result['__flow_total_log_packets'] = np.log1p(number('spkts').clip(lower=0) + number('dpkts').clip(lower=0))
    result['__flow_total_log_bytes'] = np.log1p(number('sbytes').clip(lower=0) + number('dbytes').clip(lower=0))
    if {'sttl', 'dttl'} <= set(frame):
        result['__flow_ttl_difference'] = number('sttl') - number('dttl')
    for left, right in (('proto', 'service'), ('proto', 'state'), ('service', 'state')):
        a = frame[left].fillna('__MISSING__').astype(str)
        b = frame[right].fillna('__MISSING__').astype(str)
        name = '__flow_' + left + '_' + right
        # Length prefixes preserve boundaries even for adversarial input strings.
        result[name] = a.str.len().astype(str) + ':' + a + b.str.len().astype(str) + ':' + b
        categorical.append(name)
    return result, categorical
