import click


@click.group()
def cli():
    pass


@cli.command()
def datagen_train_run():
    from burybarrel.scripts import datagen_occ, train_barrelnet
    datagen_occ.run()
    train_barrelnet.run()


@cli.command()
def run_pointnet_inference():
    from burybarrel.scripts import run_pointnet_inf
    run_pointnet_inf.run()
