#!/bin/sh
set -eu

mkdir -p "$1/events" "$1/clicks" "$1/links"
cd "$1"
awk '
function stamp(day, hour, minute, second, format) {
    if (format == 0)
        return sprintf("2026-01-%02dT%02d:%02d:%02dZ", day, hour, minute, second)
    if (format == 1)
        return sprintf("2026-01-%02dT%02d:%02d:%02d-05:00", day, hour-5, minute, second)
    return sprintf("%.0f", 1767225600000 + ((day-1)*86400 + hour*3600 + minute*60 + second)*1000)
}
function quote(value, n) { return n % 5 == 0 ? "\"" value "\"" : value }
BEGIN {
    print "{\"window_start\":\"2026-01-01T00:00:00Z\",\"window_end\":\"2026-01-26T00:00:00Z\"}" > "manifest.json"
    print "product_id,category" > "products.csv"
    print "campaign_id,channel" > "campaigns.csv"
    print "user_id_a,user_id_b" > "links/part.csv"
    for (i = 0; i < 1024; ++i) print "p" i ",category-" (i % 48) > "products.csv"
    for (i = 0; i < 96; ++i) print "c" i ",channel-" (i % 8) > "campaigns.csv"
    for (i = 0; i < 2400; ++i) print "u" i ",a" i > "links/part.csv"

    for (part = 0; part < 8; ++part) {
        ev = "events/part-" part ".csv"
        cl = "clicks/part-" part ".csv"
        print "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents" > ev
        print "click_id,user_id,campaign_id,ts" > cl
        for (j = 0; j < 32000; ++j) {
            i = part * 32000 + j
            u = int((i * 7907 % 131071) / 7)
            user = ((j % 4 == 0 && u < 2400) ? "a" : "u") u
            day = 1 + int(u / 791) % 24
            hour = 8 + u % 12
            minute = int(j / 11) % 60
            t = stamp(day, hour, minute, i % 60, i % 10 < 6 ? 0 : (i % 10 < 7 ? 1 : 2))
            ingest = stamp(day, hour, minute, i % 60, 0)
            type = i % 13 < 2 ? "purchase" : (i % 13 < 5 ? "cart" : "view")
            id = (i % 4079 == 0 ? "long-training-event-" : "e") i
            print quote(id, i) "," quote(user, j+2) "," quote(i % 211 == 0 ? "invalid" : t, i+1) "," ingest "," type ",p" (i % 1024) "," (100+i%150000) > ev
            if (i % 31 == 0) print id ",different-user," t "," stamp(day+1, hour, minute, 0, 0) ",view,p1," > ev
            if (i % 1013 == 0) print "invalid,row,\"\"" > ev
            if (j % 3 == 0) {
                click = "k" i "," user ",c" (i % 96) "," stamp(day, hour-1, minute, i % 60, i % 3 == 0 ? 2 : j % 2)
                print click > cl
                if (j % 51 == 0) print click > cl
            }
        }
        close(ev)
        close(cl)
    }
}'
