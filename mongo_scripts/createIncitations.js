// Using a HIGHLY inefficient algorithm currently! Need to optimize using GroupBy?

print('Warning: Using a HIGHLY inefficient algorithm currently! Need to optimize using GroupBy');
var maxId = db.data.count();

for (var i = 1; i <= maxId; i++) {
    var documentId = i.toString();
    db.inCitations.insert({id: documentId, citedBy: []});
}


for (i = 1; i <= maxId; i++) {
    documentId = i.toString();
    db.data.findOne({"id": documentId}).out_citations.forEach(citeId => {
        print(citeId, documentId);
        db.inCitations.update({id: citeId},
            {$push: {citedBy: documentId}}
        )
    });
}
