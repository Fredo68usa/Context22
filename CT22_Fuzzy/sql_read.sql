select * from currentsqls;
select * from similarsqls;
select curr1.SQL as Tested, curr2.SQL as against, sim.status, sim.score from similarsqls sim , currentsqls curr1, currentsqls curr2  where sim.hash=curr1.hash and sim.hashSim=curr2.hash;
select curr1.SQL as Tested, curr2.SQL as against, sim.status, sim.score from similarsqls sim , currentsqls curr1, currentsqls curr2  where sim.hash=curr1.hash and sim.hashSim=curr2.hash and sim.score > 70;
